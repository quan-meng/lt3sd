import torch
import os
import numpy as np
from tqdm import tqdm
import time
from typing import Union

from tools.logger import get_logger
from tools import scene_utils, mesh_utils, common_utils

logger = get_logger(file_name=__file__, debug="fusion")


class Scene(object):
    def __init__(self, levels, channels, scene_shape, device="cpu"):
        self.levels = levels
        self.channels = channels
        self.device = device

        self.latent_shapes, self.scene_shapes = self.init_scene_size(scene_shape)

        self.latents, self.known_masks, self.orders = {}, {}, {}
        for level in levels:
            scene_shape = self.latent_shapes[level]
            channel = self.channels[level]

            latent = torch.ones(
                (1, channel, *scene_shape), device=self.device
            )  # [1, C, H, W, D]
            known_mask = torch.zeros(
                (1, channel, *scene_shape), device=self.device
            )  # mask for known region

            self.latents[level] = latent
            self.known_masks[level] = known_mask

    def init_scene_size(self, scene_shape):
        _, vxl_size_min_i, _ = common_utils.parse_level(self.levels[-1])
        _, _, vxl_size_max_o = common_utils.parse_level(self.levels[0])
        factor_all = int(round(vxl_size_max_o / vxl_size_min_i))
        scene_shape = [x - x % factor_all for x in scene_shape]

        latent_shapes, scene_shapes = {}, {}
        for level in self.levels:
            _, vxl_size_i, vxl_size_o = common_utils.parse_level(level)
            factor_1 = int(round(vxl_size_i / vxl_size_min_i))
            factor_2 = int(round(vxl_size_o / vxl_size_i))
            latent_shapes[level] = [x // (factor_1 * factor_2) for x in scene_shape]
            scene_shapes[level] = [x // factor_1 for x in scene_shape]

        return latent_shapes, scene_shapes


def split_scene_latent(x, model, level):
    if model.start_level == level:
        return x, None
    else:
        return x[:, 1:], x[:, :1]


def concat_scene_latent(x, c):
    if c is None:
        return x
    else:
        return torch.cat((c, x), dim=1)


@torch.no_grad()
def is_fusion(
    model,
    scene: Scene,
    level: str,
    overlap: Union[float, int],
    with_chunk=False,
    with_diffusion=False,
    out_dir=None,
    **kwargs,
):
    device = model.device

    chunk_shape = model.chunk_shape[level]
    latent_shape = scene.latent_shapes[level]
    step_size = [
        x - int(x * overlap) if isinstance(overlap, float) else x - overlap
        for x in chunk_shape
    ]

    scene_size_ext = [max(x, y) for x, y in zip(latent_shape, chunk_shape)]

    infos = {}
    poses = scene_utils.get_pos_sequence(chunk_shape, step_size, scene_size_ext)
    poses = poses.to(device)
    infos.update({"chunk_num": len(poses)})
    scene_latent = scene.latents[level].clone()
    scene_latent = model.first_stage_model.normalize(scene_latent, level=level)
    scene_x, scene_c = split_scene_latent(scene_latent, model, level)
    scene_mask = split_scene_latent(scene.known_masks[level].clone(), model, level)[0]

    model_channel = scene_x.shape[1]
    latent = torch.zeros_like(scene_x)
    latent[..., : latent_shape[0], : latent_shape[1], : latent_shape[2]] = scene_x

    mask = torch.zeros_like(scene_x)
    mask[..., : latent_shape[0], : latent_shape[1], : latent_shape[2]] = scene_mask

    out_dir_l = os.path.join(out_dir, f"level_{level}") if with_chunk else out_dir
    os.makedirs(out_dir_l, exist_ok=True)

    inputs_i = {}
    for i, pos in tqdm(enumerate(poses), desc="SI Fusioning ...", total=len(poses)):
        bbox_min, bbox_max = scene_utils.pose2bbox(pos, chunk_shape)
        x0 = latent[
            ...,
            bbox_min[0] : bbox_max[0],
            bbox_min[1] : bbox_max[1],
            bbox_min[2] : bbox_max[2],
        ].clone()
        known_mask = mask[
            ...,
            bbox_min[0] : bbox_max[0],
            bbox_min[1] : bbox_max[1],
            bbox_min[2] : bbox_max[2],
        ].clone()

        shape = (1, model_channel, *chunk_shape)

        if scene_c is None:
            c = None
        else:
            c = {
                "c_concat": scene_c[
                    ...,
                    bbox_min[0] : bbox_max[0],
                    bbox_min[1] : bbox_max[1],
                    bbox_min[2] : bbox_max[2],
                ].clone()
            }

        inputs_i.update(
            {
                "x": torch.randn((1, model_channel, *chunk_shape), dtype=torch.float),
                "x0": x0,
                "mask": known_mask,
                "level": level,
                "c": c,
            }
        )

        inputs_i = common_utils.recursive_to(inputs_i, device=device)

        _, inters, inputs_i = model.sample(
            inputs_i, batch_size=1, ddim=True, shape=shape, **kwargs
        )

        latent[
            ...,
            bbox_min[0] : bbox_max[0],
            bbox_min[1] : bbox_max[1],
            bbox_min[2] : bbox_max[2],
        ] = (
            inputs_i["x"] * (1 - known_mask) + x0 * known_mask
        )
        mask[
            ...,
            bbox_min[0] : bbox_max[0],
            bbox_min[1] : bbox_max[1],
            bbox_min[2] : bbox_max[2],
        ] = 1.0

        if with_chunk:
            latent_i = concat_scene_latent(
                inputs_i["x"], c["c_concat"] if c is not None else None
            )
            latent_i = model.first_stage_model.denormalize(latent_i, level=level)
            chunk = model.first_stage_model.decode(latent_i, level=level)[0, 0]

            if i == 0:
                _, vxl_size_i, vxl_size_o = common_utils.parse_level(level)
                factor = int(round(vxl_size_o / vxl_size_i))
                volume = torch.ones([x * factor for x in latent_shape], device=device)

            bbox_min_v = [int(p * factor) for p in bbox_min]
            bbox_max_v = [int(p * factor) for p in bbox_max]
            volume[
                bbox_min_v[0] : bbox_max_v[0],
                bbox_min_v[1] : bbox_max_v[1],
                bbox_min_v[2] : bbox_max_v[2],
            ] = chunk

            vxl_size = 2.8 / (volume.shape[1] - 1)
            threshold = mesh_utils.threshold_ada(vxl_size)
            mesh = mesh_utils.volume_to_mesh(volume, threshold=threshold)
            mesh.export(os.path.join(out_dir_l, f"chunk_{i}.ply"))

        if with_diffusion:
            out_dir_c = os.path.join(out_dir_l, f"chunk_{i}")
            os.makedirs(out_dir_c, exist_ok=True)

            x_inters = inters["x_inter"]  # list
            log_inter = len(x_inters) // 5
            for j, x_inter in enumerate(x_inters):
                if j % log_inter == 0 or j == len(x_inters) - 1:
                    latent_i = x_inter * (1 - known_mask) + x0 * known_mask
                    latent_i = concat_scene_latent(
                        latent_i, c["c_concat"] if c is not None else None
                    )
                    latent_i = model.first_stage_model.denormalize(
                        latent_i, level=level
                    )  # [1, C, G1, G2, G3]
                    chunk = model.first_stage_model.decode(latent_i, level=level)[0, 0]

                    volume[
                        bbox_min_v[0] : bbox_max_v[0],
                        bbox_min_v[1] : bbox_max_v[1],
                        bbox_min_v[2] : bbox_max_v[2],
                    ] = chunk

                    vxl_size = 2.8 / (volume.shape[1] - 1)
                    threshold = mesh_utils.threshold_ada(vxl_size)
                    mesh = mesh_utils.volume_to_mesh(volume, threshold=threshold)
                    mesh.export(os.path.join(out_dir_c, f"timestep_{j}.ply"))

    assert torch.all(mask == 1.0), f"mask mean: {mask.mean()}"
    scene_x = latent[..., : latent_shape[0], : latent_shape[1], : latent_shape[2]]
    scene_latent = concat_scene_latent(scene_x, scene_c)
    scene_latent = model.first_stage_model.denormalize(
        scene_latent, level=level
    )  # [1, C, G1, G2, G3]

    return scene_latent, infos


@torch.no_grad()
def multi_fusion(
    model,
    scene: Scene,
    level: str,
    overlap: Union[float, int],
    log_every_t=20,
    out_dir=None,
    with_chunk=False,
    with_diffusion=False,
    ddim_steps: int = 200,
    ddim_eta: float = 1.0,
    mini_batch: int = 64,
    **kwargs,
):
    device = model.device

    chunk_shape = model.chunk_shape[level]
    latent_shape = scene.latent_shapes[level]
    step_size = [
        x - int(x * overlap) if isinstance(overlap, float) else x - overlap
        for x in chunk_shape
    ]

    model.make_schedule(
        ddim_num_steps=ddim_steps,
        ddim_eta=ddim_eta,
        verbose=False,
        device=device,
    )
    time_range = np.flip(model.ddim_timesteps)

    infos = {}
    poses = scene_utils.get_poses(chunk_shape, step_size, latent_shape).to(device)
    infos.update({"chunk_num": len(poses)})

    scene_latent = scene.latents[level].clone()
    scene_latent = model.first_stage_model.normalize(scene_latent, level=level)
    scene_x, scene_c = split_scene_latent(scene_latent, model, level)

    model_channel = scene_x.shape[1]

    scene_size_ext = [max(x, y) for x, y in zip(latent_shape, chunk_shape)]
    noise = torch.randn(
        (1, model_channel, *scene_size_ext), device=device, dtype=torch.float
    )
    count = torch.zeros((1, 1, *scene_size_ext), device=device, dtype=torch.float)
    value = torch.zeros(
        (1, model_channel, *scene_size_ext), device=device, dtype=torch.float
    )

    out_dir_l = os.path.join(out_dir, f"level_{level}") if with_chunk else out_dir
    os.makedirs(out_dir_l, exist_ok=True)

    total_steps = time_range.shape[0]
    for i, t in tqdm(
        enumerate(time_range), total=len(time_range), desc="Multi Fusioning ..."
    ):
        count.zero_()
        value.zero_()

        if torch.all(scene.known_masks[level][0, 0] > 0):
            cs = []
            for _, pos in enumerate(poses):
                bbox_min, bbox_max = scene_utils.pose2bbox(pos, chunk_shape)
                c_i = scene_c[
                    ...,
                    bbox_min[0] : bbox_max[0],
                    bbox_min[1] : bbox_max[1],
                    bbox_min[2] : bbox_max[2],
                ].clone()
                cs.append(c_i)
            cs = torch.cat(cs)
        else:
            cs = None

        xs = []
        for _, pos in enumerate(poses):
            bbox_min, bbox_max = scene_utils.pose2bbox(pos, chunk_shape)
            x_i = noise[
                ...,
                bbox_min[0] : bbox_max[0],
                bbox_min[1] : bbox_max[1],
                bbox_min[2] : bbox_max[2],
            ].clone()
            xs.append(x_i)
        xs = torch.cat(xs)  # [P, C, g, g, g]

        ts = torch.full((len(poses),), t, device=device, dtype=torch.long)
        outs = []

        for j in range(0, len(poses), mini_batch):
            x_j = xs[j : j + mini_batch]
            c = {"c_concat": cs[j : j + mini_batch]} if cs is not None else None

            inputs_j = {"x": x_j, "level": level, "c": c}
            values_j, _ = model.p_sample_ddim(
                inputs_j,
                ts[j : j + mini_batch],
                index=total_steps - i - 1,
                **kwargs,
            )

            outs.append(values_j.to(device))
        outs = torch.cat(outs)  # [P, C, g, g, g]

        for k in range(len(poses)):
            bbox_min, bbox_max = scene_utils.pose2bbox(poses[k], chunk_shape)
            value[
                ...,
                bbox_min[0] : bbox_max[0],
                bbox_min[1] : bbox_max[1],
                bbox_min[2] : bbox_max[2],
            ] += outs[[k]]
            count[
                ...,
                bbox_min[0] : bbox_max[0],
                bbox_min[1] : bbox_max[1],
                bbox_min[2] : bbox_max[2],
            ] += 1.0

        noise = torch.where(count > 0, value / count, value)

        if with_diffusion and (i % log_every_t == 0 or i == len(time_range) - 1):
            noise_t = (
                noise[..., : latent_shape[0], : latent_shape[1], : latent_shape[2]]
                .clone()
                .to(device)
            )
            latent_t = concat_scene_latent(noise_t, scene_c if c is not None else None)
            latent_t = model.first_stage_model.denormalize(
                latent_t, level=level
            )  # [1, C, G1, G2, G3]
            volume = model.first_stage_model.decode(latent_t, level=level)[0, 0]

            vxl_size = 2.8 / (volume.shape[1] - 1)
            threshold = mesh_utils.threshold_ada(vxl_size)
            mesh = mesh_utils.volume_to_mesh(volume, threshold=threshold)
            mesh.export(os.path.join(out_dir_l, f"timestep_{i}.ply"))

    scene_x = noise[..., : latent_shape[0], : latent_shape[1], : latent_shape[2]].to(
        device
    )
    scene_latent = concat_scene_latent(scene_x, scene_c)
    scene_latent = model.first_stage_model.denormalize(scene_latent, level=level)
    scene_latent = scene_latent.to(device)  # [1, C, G1, G2, G3]

    return scene_latent, infos


def fusion(model, scene, out_dir="./", n_repeat=1, **kwargs):
    for r in range(n_repeat):
        out_dir_r = os.path.join(out_dir, f"repeat_{r}") if n_repeat > 1 else out_dir
        os.makedirs(out_dir_r, exist_ok=True)

        levels = model.levels
        for idx in range(len(levels)):
            level = levels[idx]
            time_curr = time.time()

            kwargs.update(
                {"model": model, "scene": scene, "out_dir": out_dir_r, "level": level}
            )

            fusioner = "is_fusion" if level == levels[0] else "multi_fusion"
            # fusioner = "multi_fusion"

            scene_latent, infos = globals()[fusioner](**kwargs)
            scene.latents[level] = scene_latent

            time_delta = time.time() - time_curr
            logger.info(
                f'Level: {level}, Time: {time_delta:.2f}, Chunks: {infos["chunk_num"]}, Fusioner: {fusioner}'
            )

            volume = model.first_stage_model.decode(scene_latent, level=level)[0, 0]
            np.save(
                os.path.join(out_dir_r, f"volume_{level}.npy"), volume.cpu().numpy()
            )
            logger.debug(f"volume, min: {volume.min():.2f}, max: {volume.max():.2f}")

            mode, vxl_size_i, _ = common_utils.parse_level(level)
            threshold = common_utils.ada_threshold(vxl_size_i, mode, factor=1.5)
            threshold = min(model.first_stage_model.truncation * 0.8, threshold)
            mesh = mesh_utils.volume_to_mesh(volume, threshold)
            mesh.export(os.path.join(out_dir_r, f"level_{level}.ply"))
            
            if level != levels[-1]:
                next_level = levels[idx + 1]
                volume = torch.clamp(volume, min=0.0)
                scene.latents[next_level][0, 0] = volume
                scene.known_masks[next_level][0, 0] = 1.0
