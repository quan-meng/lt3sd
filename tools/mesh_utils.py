import torch
import trimesh
import numpy as np
from skimage import measure


def threshold_ada(voxel_size, factor=1.5, min_val=0.01, max_val=0.07):
    threshold = min(max(voxel_size * factor, min_val), max_val)
    return threshold


def volume_to_mesh(volume, threshold=0.0):
    """
    Args:
        volume: [G, G, G]
    """
    volume = volume.cpu().numpy() if isinstance(volume, torch.Tensor) else volume
    assert volume.ndim == 3, f"Input volume of shape {volume.shape} is wrong!"

    try:
        vertices, triangles, _, _ = measure.marching_cubes(
            volume, level=threshold, gradient_direction="descent"
        )
    except:
        vertices = triangles = np.zeros([1, 3])

    vertices = vertices / (volume.shape[1] - 1.0)  # Keep y axis as [0, 1]
    vertices = vertices * 2 - 1.0  # Normalize to [-1, 1]

    mesh = trimesh.Trimesh(vertices, triangles, process=False)

    return mesh


def merge_meshes(mesh_list, span=None, dim=-1, inverse=False):
    if inverse:
        mesh_list = mesh_list[::-1]

    # Initialize arrays for vertices, faces, and normals
    all_vertices = []
    all_faces = []
    all_normals = []

    # Initialize array for colors if the first mesh has vertex colors
    all_colors = []
    has_colors = hasattr(mesh_list[0].visual, "vertex_colors")

    vertex_count = 0
    for i, mesh in enumerate(mesh_list):
        # Apply span if provided
        if span is not None:
            spans_i = np.eye(3)[dim] * span * i
            v = mesh.vertices + spans_i
        else:
            v = mesh.vertices

        # Concatenate vertices, faces, and normals
        all_vertices.append(v)
        all_faces.append(mesh.faces + vertex_count)
        all_normals.append(mesh.face_normals)

        # Concatenate colors if present
        if has_colors:
            mesh_colors = getattr(mesh.visual, "vertex_colors", None)
            if mesh_colors is not None:
                all_colors.extend(mesh_colors)

        vertex_count += len(mesh.vertices)

    # Create the merged mesh
    merged_mesh = trimesh.Trimesh(
        vertices=np.vstack(all_vertices),
        faces=np.vstack(all_faces),
        face_normals=np.vstack(all_normals),
        process=False,
    )

    # Assign concatenated colors to the merged mesh if there were any
    if has_colors and all_colors:
        merged_mesh.visual.vertex_colors = np.array(all_colors)

    return merged_mesh
