import torch.nn.functional as F
import torch
import random
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from tools.common_utils import int2tuple


def pose2bbox(pose, chunk_shape) -> tuple[tuple[int], tuple[int]]:
    """
    Args:
        poses: [3]
    """
    chunk_shape = int2tuple(chunk_shape, 3)

    bbox_min = [int(x - y // 2) for x, y in zip(pose, chunk_shape)]
    bbox_max = [int(x + y // 2) for x, y in zip(pose, chunk_shape)]

    return bbox_min, bbox_max


def get_adjacent_points(pts, step_size, chunk_shape, bbox_min, bbox_max):
    """
    Args:
        pts: [B, 3]
    """
    step_size = int2tuple(step_size, 3)

    # Get adjacent points
    i, j, k = torch.split(pts, 1, dim=-1)  # [B, 1]
    i_adj = torch.stack(
        [i + step_size[0], i - step_size[0], i, i, i, i], dim=-1
    )  # [B, 6]
    j_adj = torch.stack(
        [j, j, j + step_size[1], j - step_size[1], j, j], dim=-1
    )  # [B, 6]
    k_adj = torch.stack(
        [k, k, k, k, k + step_size[2], k - step_size[2]], dim=-1
    )  # [B, 6]
    adjacent_points = torch.stack((i_adj, j_adj, k_adj), dim=-1)  # [B, 6, 3]
    adjacent_points = adjacent_points.view(-1, 3)  # [B * 6, 3]

    # Get the scene inner and outer bounding box
    scene_bbox_inner = torch.cat(
        [bbox_min + chunk_shape // 2, bbox_max - chunk_shape // 2]
    )  # [6]
    scene_bbox_outer = torch.cat(
        [bbox_min - chunk_shape // 2 + 1, bbox_max + chunk_shape // 2]
    )  # [6]

    # Filter out points outside the outer bounding box
    valid_points_mask = torch.all(
        adjacent_points >= scene_bbox_outer[None, :3], dim=-1
    ) & torch.all(
        adjacent_points < scene_bbox_outer[None, 3:], dim=-1
    )  # [B * 6]
    adjacent_points = adjacent_points[valid_points_mask]  # [N, 3]

    # Clamp the points to the inner bounding box if chunk_shape is smaller than the scene size
    for dim in range(3):
        if scene_bbox_inner[dim] <= scene_bbox_inner[dim + 3]:
            adjacent_points[:, dim] = torch.clamp(
                adjacent_points[:, dim],
                min=scene_bbox_inner[dim],
                max=scene_bbox_inner[dim + 3],
            )

    # Remove duplicate points
    adjacent_points, counts = torch.unique(adjacent_points, dim=0, return_counts=True)

    # Sort the points by the duplicated counts from high to low
    _, sorted_indices = torch.sort(counts, descending=True)
    adjacent_points = adjacent_points[sorted_indices]

    # Remove the original point
    matches = torch.all(adjacent_points[:, None, :] == pts, dim=2)
    adjacent_points = adjacent_points[~torch.any(matches, dim=1)]

    return adjacent_points


def get_poses(chunk_shape, step_size, scene_shape):
    if chunk_shape[0] >= scene_shape[0]:
        xs = [chunk_shape[0] // 2]
    else:  # append the last chunk if the last chunk does not cover the whole scene
        xs = list(
            range(
                chunk_shape[0] // 2, scene_shape[0] - chunk_shape[0] // 2, step_size[0]
            )
        )
        if xs[-1] != scene_shape[0] - chunk_shape[0] // 2:
            xs += [scene_shape[0] - chunk_shape[0] // 2]

    if chunk_shape[1] >= scene_shape[1]:
        ys = [chunk_shape[1] // 2]
    else:
        ys = list(
            range(
                chunk_shape[1] // 2, scene_shape[1] - chunk_shape[1] // 2, step_size[1]
            )
        )
        if ys[-1] != scene_shape[1] - chunk_shape[1] // 2:
            ys += [scene_shape[1] - chunk_shape[1] // 2]

    if chunk_shape[2] >= scene_shape[2]:
        zs = [chunk_shape[2] // 2]
    else:
        zs = list(
            range(
                chunk_shape[2] // 2, scene_shape[2] - chunk_shape[2] // 2, step_size[2]
            )
        )
        if zs[-1] != scene_shape[2] - chunk_shape[2] // 2:
            zs += [scene_shape[2] - chunk_shape[2] // 2]

    poses = torch.stack(
        torch.meshgrid(
            torch.tensor(xs), torch.tensor(ys), torch.tensor(zs), indexing="ij"
        ),
        dim=-1,
    )  # [X, Y, Z, 3]

    return poses.reshape(-1, 3)


def get_pos_sequence(chunk_shape, step_size, scene_shape, init_bbox=None):
    scene_shape = torch.tensor(scene_shape)

    if init_bbox is None:
        all_points = torch.tensor([x // 2 for x in chunk_shape])[None]
    else:
        init_bbox[:3] = torch.clamp(init_bbox[:3], min=chunk_shape // 2)
        init_bbox[3:] = torch.clamp(init_bbox[3:], max=scene_shape - chunk_shape // 2)

        # sample a random pts in bbox
        all_points = torch.stack(
            [
                torch.randint(init_bbox[0], init_bbox[3], (1,)),
                torch.randint(init_bbox[1], init_bbox[4], (1,)),
                torch.randint(init_bbox[2], init_bbox[5], (1,)),
            ],
            dim=-1,
        )
        all_points = all_points[None]

    mask = torch.zeros((1, 1, *(scene_shape.tolist())))
    bbox_min, bbox_max = pose2bbox(all_points[0], chunk_shape)
    mask[
        ...,
        bbox_min[0] : bbox_max[0],
        bbox_min[1] : bbox_max[1],
        bbox_min[2] : bbox_max[2],
    ] = 1.0

    while torch.any(mask == 0):
        # Find the adjacent points to the initial set
        adjacent_points = get_adjacent_points(
            all_points,
            step_size,
            torch.tensor(chunk_shape),
            torch.zeros(
                3,
            ),
            scene_shape,
        ).long()

        # Filter out the points that are already in set A or set B
        for pts in adjacent_points:
            bbox_min, bbox_max = pose2bbox(pts, chunk_shape)
            crop = mask[
                ...,
                bbox_min[0] : bbox_max[0],
                bbox_min[1] : bbox_max[1],
                bbox_min[2] : bbox_max[2],
            ]
            if torch.any(crop == 0):
                all_points = torch.cat((all_points, pts[None]))
                mask[
                    ...,
                    bbox_min[0] : bbox_max[0],
                    bbox_min[1] : bbox_max[1],
                    bbox_min[2] : bbox_max[2],
                ] = 1.0
    return all_points


def crop_with_bbox(x, bbox_min, bbox_max, pad_value=0.0):
    """
    Args:
        x: [G, G, G]
        bbox_min: [3]
        bbox_max: [3]
    Returns:
        out: [g1, g2, g3]
    """
    scene_shape = torch.tensor(x.shape[-3:]).long()  # [3]

    bbox_min_clip = torch.clamp(bbox_min, min=0)
    bbox_max_clip = torch.clamp(bbox_max, max=scene_shape)

    x = x[
        ...,
        bbox_min_clip[0] : bbox_max_clip[0],
        bbox_min_clip[1] : bbox_max_clip[1],
        bbox_min_clip[2] : bbox_max_clip[2],
    ]
    x = (
        torch.from_numpy(x) if not isinstance(x, torch.Tensor) else x
    )  # load with mmap_mode

    pad = (bbox_max - bbox_min) - torch.tensor(x.shape[-3:])
    x = x[None, None]

    mask = F.pad(
        torch.ones_like(x[:, :1]),
        pad=(0, pad[2], 0, pad[1], 0, pad[0]),
        mode="constant",
        value=0.0,
    )
    x = F.pad(x, pad=(0, pad[2], 0, pad[1], 0, pad[0]), mode="constant", value=0.0)

    x = x * mask + pad_value * (1 - mask)

    return x[0, 0], mask


def sample_random_chunk(scene_shape, chunk_shape):
    def func(scene_length, chunk_length):
        if scene_length > chunk_length:
            return random.randrange(
                chunk_length // 2, scene_length - chunk_length // 2 + 1, step=1
            )
        else:
            return chunk_length // 2

    return list(map(func, scene_shape, chunk_shape))


def sample_chunk_in_bbox(scene_shape, chunk_shape, bboxes):
    a_mins = [x // 2 for x in chunk_shape]
    a_maxs = [x - y // 2 for x, y in zip(scene_shape, chunk_shape)]

    if chunk_shape[0] < scene_shape[0]:
        bboxes[..., 0] = np.clip(bboxes[..., 0], a_min=a_mins[0], a_max=a_maxs[0])
    else:
        bboxes[..., 0] = a_mins[0]

    if chunk_shape[1] < scene_shape[1]:
        bboxes[..., 1] = np.clip(bboxes[..., 1], a_min=a_mins[1], a_max=a_maxs[1])
    else:
        bboxes[..., 1] = a_mins[1]

    if chunk_shape[2] < scene_shape[2]:
        bboxes[..., 2] = np.clip(bboxes[..., 2], a_min=a_mins[2], a_max=a_maxs[2])
    else:
        bboxes[..., 2] = a_mins[2]

    bbox = bboxes[random.randint(0, len(bboxes) - 1)]  # [2, 3]
    pose = [random.randint(bbox[0, i], bbox[1, i]) for i in range(3)]

    return pose
