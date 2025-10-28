import torch
import torchvision
import trimesh
import numpy as np

from .render_utils import render_mesh, mesh_to_central_poses


def mesh_to_pclouds(in_path, out_path, num_samples=8192):
    mesh = trimesh.load(in_path)

    points, _ = trimesh.sample.sample_surface(mesh, num_samples)
    trimesh.points.PointCloud(points).export(out_path, file_type="ply")


def mesh_to_images(
    mesh_path, out_path, num_poses=20, render_resolution=299, intensity=1.0
):
    mesh = trimesh.load(mesh_path)

    bboxes = mesh.bounding_box.bounds
    bbox_min = bboxes[0]
    bbox_max = bboxes[1]

    # move bbox_min to origin
    mesh.apply_translation(-bbox_min)  # [0, bbox_max]

    # normalize y to [0, 1]
    mesh.vertices = mesh.vertices / (bbox_max[1] - bbox_min[1])

    # normalize y to [-1, 1]
    mesh.vertices = mesh.vertices * 2 - 1

    poses = mesh_to_central_poses(mesh, num_poses)  # [N, 4, 4]

    # remove_colors:
    mesh.visual = trimesh.visual.ColorVisuals()
    mesh.visual.vertex_colors = np.ones_like(mesh.vertices)

    for j, camera_pose in enumerate(poses):
        image = (
            render_mesh(
                mesh,
                camera_pose=camera_pose,
                resolution=render_resolution,
                intensity=intensity,
            )
            / 255
        )
        torchvision.utils.save_image(
            torch.from_numpy(image.copy()).permute(2, 0, 1), out_path + f"_{j:02d}.png"
        )
