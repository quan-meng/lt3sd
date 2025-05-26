import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

from tools.common_utils import int2tuple


def sliding_windows(x, func, out_dim, chunk_shape, overlap=0, factor=2, **kwargs):
    """
    Args:
        x: [B, C, G1, G2, G3]
    Returns:
        out: [B, C, g1, g2, g3]
    """
    B, _, *scene_size = x.shape
    device = x.device

    chunk_shape = np.array(int2tuple(chunk_shape, 3)).astype(int)
    overlap = np.array(int2tuple(overlap, 3)).astype(int)
    scene_size = np.array(scene_size).astype(int)

    if any(scene_size < chunk_shape):
        return func(x, **kwargs)
    else:
        d_grid, h_grid, w_grid = np.meshgrid(
            list(range(0, scene_size[0] - chunk_shape[0], chunk_shape[0]))
            + [scene_size[0] - chunk_shape[0]],
            list(range(0, scene_size[1] - chunk_shape[1], chunk_shape[1]))
            + [scene_size[1] - chunk_shape[1]],
            list(range(0, scene_size[2] - chunk_shape[2], chunk_shape[2]))
            + [scene_size[2] - chunk_shape[2]],
            indexing="ij",
        )
        p_bboxes = np.stack((d_grid, h_grid, w_grid), axis=-1)  # [g1, g2, g3, 3]
        p_bboxes = rearrange(p_bboxes, "g1 g2 g3 c -> (g1 g2 g3) c")  # [N, 3]
        p_bboxes = np.concatenate(
            (p_bboxes, p_bboxes + chunk_shape), axis=-1
        )  # [g1, g2, g3, 6]

        # Calculate window bounding boxes for all chunks
        w_bboxes_l = np.clip(p_bboxes[:, :3] - overlap, a_min=0, a_max=None)  # [N, 3]
        w_bboxes_r = np.clip(
            p_bboxes[:, 3:] + overlap, a_min=None, a_max=scene_size
        )  # [N, 3]
        w_bboxes = np.concatenate([w_bboxes_l, w_bboxes_r], axis=-1)  # [N, 6]

        c_bbox_l = np.where(w_bboxes_l[:, :3] == 0, p_bboxes[:, :3], overlap)  # [N, 3]
        c_bboxes = np.concatenate([c_bbox_l, c_bbox_l + chunk_shape], axis=-1)  # [N, 6]

        output = torch.empty(
            (B, out_dim, *((scene_size * factor).astype(int)))
        )  # [B, C, g1, g2, g3]
        p_bboxes = (p_bboxes * factor).astype(int)  # [g1, g2, g3, 3]
        c_bboxes = (c_bboxes * factor).astype(int)  # [N, 6]

        for i, (w_bbox, c_bbox, p_bbox) in enumerate(zip(w_bboxes, c_bboxes, p_bboxes)):
            d_l, h_l, w_l, d_r, h_r, w_r = w_bbox
            patch = x[..., d_l:d_r, h_l:h_r, w_l:w_r]
            patch = func(patch, **kwargs)
            patch = patch[
                ..., c_bbox[0] : c_bbox[3], c_bbox[1] : c_bbox[4], c_bbox[2] : c_bbox[5]
            ]
            output[
                ..., p_bbox[0] : p_bbox[3], p_bbox[1] : p_bbox[4], p_bbox[2] : p_bbox[5]
            ] = patch.cpu()

        return output.to(device)


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv, out_channels=None):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels

        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv3d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv3d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv3d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = torch.nn.Conv3d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, *shape = q.shape
        q = q.reshape(b, c, math.prod(shape))
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, math.prod(shape))  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, math.prod(shape))
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, *shape)

        h_ = self.proj_out(h_)

        return x + h_


def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none"], f"attn_type {attn_type} unknown"
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    else:
        return nn.Identity(in_channels)


class Encoder(nn.Module):
    def __init__(
        self,
        factor,
        z_channels,
        in_channels,
        ch,
        num_res_blocks,
        attn_resolutions,
        ch_mult=(1, 1, 1, 1, 1),
        dropout=0.0,
        resamp_with_conv=True,
        double_z=True,
        attn_type="vanilla",
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_res_blocks = num_res_blocks
        self.factor = factor
        self.in_channels = in_channels
        self.z_channels = z_channels
        self.out_channels = 2 * z_channels if double_z else z_channels

        # downsampling
        self.conv_in = torch.nn.Conv3d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        self.time = int(math.log2(factor))
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.time + 1):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if (2**i_level) in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.time - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv3d(
            block_in, self.out_channels, kernel_size=3, stride=1, padding=1
        )

    def forward_window(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.time + 1):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.time - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

    def forward(self, x, chunk_shape=[32, 16, 32]):
        chunk_shape = [x * int(self.factor) for x in chunk_shape]
        overlap = [x // 2 for x in chunk_shape]
        out = sliding_windows(
            x,
            self.forward_window,
            out_dim=self.out_channels,
            chunk_shape=chunk_shape,
            overlap=overlap,
            factor=1.0 / self.factor,
        )
        mean, logvar = out.chunk(2, dim=1)

        return mean, logvar


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        ch,
        out_ch,
        num_res_blocks,
        factor,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        give_pre_end=False,
        tanh_out=False,
        ch_mult=(1, 1, 1, 1, 1),
        attn_type="vanilla",
        **ignorekwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.factor = factor
        self.z_channels = in_channels
        self.out_channels = out_ch
        self.num_res_blocks = num_res_blocks
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.time = int(math.log2(factor))

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.time - 1]

        self.conv_in = torch.nn.Conv3d(
            in_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.time + 1)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if (2**i_level) in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv3d(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )

    def forward_window(self, z):
        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.time + 1)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h

    def forward(self, z, overlap=4, chunk_shape=[32, 16, 32]):
        overlap = [x // 2 for x in chunk_shape]
        return sliding_windows(
            z,
            self.forward_window,
            out_dim=self.out_channels,
            chunk_shape=chunk_shape,
            overlap=overlap,
            factor=self.factor,
        )
