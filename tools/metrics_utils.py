import os
import json
import math
import numpy as np
import trimesh
from typing import Optional, Tuple
import pyrender
from pyrender import (
    DirectionalLight,
    SpotLight,
    PointLight,
    OffscreenRenderer,
    RenderFlags,
)

from pytorch_fid.fid_score import calculate_fid_given_paths

from toolbox.log.logger import Logger

logger = Logger(file_name=os.path.basename(__file__), level="INFO")


def look_at(
    eye: np.ndarray,  # shape: [N, 3], camera location
    target: np.ndarray,  # shape: [N, 3], target position
    up: np.ndarray = np.array([0, 0, 1]),  # shape: [3], up vector
    system: str = "opengl",  # camera coordinate system: "blender", "opencv", or "opengl"
) -> np.ndarray:  # returns: [N, 4, 4] camera-to-world transformation matrix
    """Compute batch-wise look-at transformation matrices.

    Args:
        eye: Camera locations of shape [N, 3]
        target: Target positions of shape [N, 3]
        up: Up vector of shape [3], defaults to [0, 0, 1]
        system: Camera coordinate system, one of:
            - "blender": RIGHT, UP, BACK
            - "opencv": RIGHT, DOWN, FRONT
            - "opengl": RIGHT, UP, BACK (default)

    Returns:
        World-to-camera transformation matrices of shape [N, 4, 4]
    """
    # Compute the forward vector from target to eye
    f = eye - target
    f /= np.linalg.norm(f, axis=1, keepdims=True)

    # Compute the right vector
    r = np.cross(up, f)
    r /= np.linalg.norm(r, axis=1, keepdims=True)

    # Recompute the up vector
    u = np.cross(f, r)
    u /= np.linalg.norm(u, axis=1, keepdims=True)

    # Create a 4x4 look-at matrix
    lookat_matrix = np.eye(4)[None].repeat(eye.shape[0], axis=0)
    lookat_matrix[:, :3, 0] = r
    lookat_matrix[:, :3, 1] = u
    lookat_matrix[:, :3, 2] = f
    lookat_matrix[:, :3, 3] = eye

    # Adjust for different camera systems
    if system.lower() == "opencv":
        # OpenCV: RIGHT, DOWN, FRONT
        lookat_matrix[:, 1:3, :3] *= -1
    elif system.lower() == "opengl":
        # OpenGL: RIGHT, UP, BACK
        pass
    else:
        raise ValueError(f"Unknown camera system: {system}")

    return lookat_matrix


def cam_center_horiz_rot(
    num_views: int = 20,
    bboxes: Tuple[float, ...] = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
    up=np.array([0.0, 1.0, 0.0]),
):
    """
    Create camera poses around the center of the bounding box with horizontal rotation.
    Args:
        bboxes: [xmin, ymin, zmin, xmax, ymax, zmax]
    Returns:
        camera_poses: [N, 4, 4]
    """
    bboxes = np.array(bboxes).reshape(2, 3)  # [2, 3]
    center = np.mean(bboxes, axis=0)  # [3]

    eye = center[None].repeat(num_views, axis=0)  # [N, 3]
    radius = np.linalg.norm(bboxes[1] - bboxes[0]) * 1.5
    targets = np.array(
        [
            [
                center[0] + math.cos(angle) * radius,
                center[1],
                center[2] + math.sin(angle) * radius,
            ]
            for angle in np.linspace(0, 2 * np.pi, num_views)
        ]
    )  # [N, 3]

    camera_poses = look_at(eye, targets, up)  # [N, 4, 4]

    return camera_poses


def reorder_faces_for_camera(
    mesh: trimesh.Trimesh, camera_position: np.ndarray
) -> trimesh.Trimesh:
    """
    Reorder faces in a mesh to prevent back-face culling based on camera position.

    Args:
        mesh: trimesh object
        camera_position: [3] array, camera position in world coordinates

    Returns:
        trimesh.Trimesh: New mesh with reordered faces
    """
    vertices = mesh.vertices
    faces = mesh.faces.copy()  # Create a copy to avoid modifying original mesh

    # Calculate face centers
    face_centers = vertices[faces].mean(axis=1)

    # Calculate vectors from camera to face centers
    camera_to_face = face_centers - camera_position

    # Calculate face normals
    face_normals = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    )

    # Add small epsilon to avoid division by zero
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    eps = 1e-10
    norms = np.maximum(norms, eps)  # Ensure no zeros in denominator
    face_normals = face_normals / norms

    # Calculate dot product between camera rays and face normals
    dots = np.sum(camera_to_face * face_normals, axis=1)

    # Flip faces where dot product is positive (facing away from camera)
    flip_mask = dots > 0
    faces[flip_mask] = faces[flip_mask][:, ::-1]

    # Collect all visual attributes
    visual_kwargs = {}
    if mesh.visual.kind == "face":
        face_colors = mesh.visual.face_colors.copy()
        if flip_mask.any():
            face_colors[flip_mask] = face_colors[flip_mask]  # No need to reverse colors
        visual_kwargs["face_colors"] = face_colors
    elif mesh.visual.kind == "vertex":
        visual_kwargs["vertex_colors"] = mesh.visual.vertex_colors

    # Create new mesh with reordered faces and preserved colors
    return trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,  # Prevent trimesh from processing/changing the mesh
        visual=trimesh.visual.ColorVisuals(**visual_kwargs),
    )


def init_light(scene, camera_pose, intensity=6.0) -> None:
    direc_l = DirectionalLight(color=np.ones(3), intensity=intensity)
    spot_l = SpotLight(
        color=np.ones(3),
        intensity=intensity,
        innerConeAngle=np.pi / 16,
        outerConeAngle=np.pi / 6,
    )
    point_l = PointLight(color=np.ones(3), intensity=2 * intensity)
    direc_l_node = scene.add(direc_l, pose=camera_pose)
    point_l_node = scene.add(point_l, pose=camera_pose)
    spot_l_node = scene.add(spot_l, pose=camera_pose)


def render_mesh(
    mesh: trimesh.Trimesh,
    camera_pose,
    resolution: int = 300,
    light: bool = True,
    intensity: float = 3.0,
    fov: float = np.pi / 2.0,
    aspectRatio: float = 1.0,
    bg_color=None,
    use_cpu: bool = False,
):
    """
    Render a mesh with a given camera pose.
    Args:
        mesh: trimesh object
        camera_pose: 4x4 camera to world matrix
        resolution: int
        light: bool
        intensity: float
        bg_color: None or [3]
        use_cpu: bool, force CPU rendering if True
    Return:
        - color: [H, W, 3], float, [0, 1]
        - depth: [H, W], float, [0, ~]
    """
    # Reorder faces to prevent back-face culling
    camera_position = camera_pose[:3, 3]
    mesh_reordered = reorder_faces_for_camera(mesh, camera_position)

    # Convert to pyrender mesh
    mesh = pyrender.Mesh.from_trimesh(mesh_reordered, smooth=False)

    # renderer
    resolution = (resolution, resolution) if isinstance(resolution, int) else resolution

    # Set rendering platform before creating renderer
    if use_cpu:
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
        logger.warning("Using CPU rendering")
    else:
        os.environ["PYOPENGL_PLATFORM"] = "egl"

    # try:
    r = OffscreenRenderer(resolution[0], resolution[1])

    scene = pyrender.Scene(bg_color=bg_color)
    scene.add(mesh)

    camera = pyrender.PerspectiveCamera(yfov=fov, aspectRatio=aspectRatio)
    camera = scene.add(camera, pose=camera_pose)

    if light:
        init_light(scene, camera_pose, intensity=intensity)

    scene.set_pose(camera, camera_pose)

    render_flags = RenderFlags.NONE
    if light:
        render_flags |= RenderFlags.ALL_SOLID | RenderFlags.FACE_NORMALS
    else:
        render_flags |= RenderFlags.FLAT

    color, depth = r.render(scene, flags=render_flags)
    r.delete()

    color = color.astype(np.float32) / 255.0

    return color, depth


def compute_fid_2d(
    src_dir: str,
    tgt_dir: str,
    batch_size: int = 50,
    device: str = "cuda",
    dims: int = 2048,
    num_workers: int = 1,
    out_path: Optional[str] = None,
):
    """Compute FID between two directories of images."""
    fid = calculate_fid_given_paths(
        [src_dir, tgt_dir], batch_size, device, dims, num_workers
    )

    results = {"FID-2D": fid}

    if out_path is not None:
        with open(out_path, "w") as file:
            file.write(json.dumps(results, indent=4))

    return fid
