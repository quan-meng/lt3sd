import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
import matplotlib

matplotlib.use("Agg")
import pyglet

pyglet.options["shadow_window"] = False

import torch
import torchvision
from PIL import Image
import numpy as np
import pyrender
import trimesh
from pyrender import (
    DirectionalLight,
    SpotLight,
    PointLight,
)


# A batch-wise version of lookat function target size of [N, 3], eye: [3], up: [3]
def look_at(eye, target, up):
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
    lookat_matrix[:, 0, :3] = r
    lookat_matrix[:, 1, :3] = u
    lookat_matrix[:, 2, :3] = f
    lookat_matrix[:, :3, 3] = eye

    return lookat_matrix


def get_centeral_poses(N, bboxes, up=np.array([0.0, 1.0, 0.0]), radius=10):
    # Camera position at the center of the bounding box
    eye = np.array(
        [
            (bboxes[0, 0] + bboxes[1, 0]) / 2.0,
            (bboxes[0, 1] + bboxes[1, 1]) / 2.0,
            (bboxes[0, 2] + bboxes[1, 2]) / 2.0,
        ]
    )  # [3]

    # angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles = np.random.uniform(0, 2 * np.pi, N)
    x = radius * np.cos(angles)
    z = radius * np.sin(angles)
    targets = np.stack([x, eye[1] * np.ones(N), z], axis=1)  # [N, 3]
    eye = eye[None].repeat(N, axis=0)  # [N, 3]
    camera_poses = look_at(eye, targets, up)  # [N, 4, 4]

    return camera_poses


def mesh_to_central_poses(mesh, num_poses=20):
    """
    Generate camera poses located at the center of the mesh bounding box, and generate num_poses of looking around camera poses
    """
    bboxes = mesh.bounding_box.bounds
    pose = get_centeral_poses(num_poses, bboxes, up=np.array([0.0, 1.0, 0.0]))

    return pose


def render_mesh(
    mesh,
    camera_pose,
    resolution=1024,
    intensity=1.0,
    background=None,
    scale=1,
    no_fix_normal=True,
):
    render = Render(
        size=resolution,
        camera_pose=camera_pose,
        background=background,
        intensity=intensity,
    )

    rendered_image, _ = render.render(
        path=None, clean=True, mesh=mesh, only_render_images=no_fix_normal
    )

    return rendered_image


def render_pclouds(
    pclouds, camera_pose, resolution=1024, intensity=1.0, background=None
):
    render = Render(
        size=resolution,
        camera_pose=camera_pose,
        background=background,
        intensity=intensity,
    )

    rendered_image, _ = render.render_pointclouds(mesh=pclouds)

    return rendered_image


def render_for_fid(inputs, out_dir, render_resolution=299, intensity=1.0):
    mesh_path, camera_poses = inputs

    # if error occurs, print the file name, and continue
    try:
        mesh = trimesh.load(mesh_path)
    except:
        print(f"Error loading mesh: {mesh_path}")
        return

    file_name = os.path.basename(mesh_path).split(".")[0]

    for j, camera_pose in enumerate(camera_poses):
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
            torch.from_numpy(image.copy()).permute(2, 0, 1),
            os.path.join(out_dir, f"{file_name}_{j}.png"),
        )


def scale_to_unit_sphere(mesh, evaluate_metric=False):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = mesh.vertices - mesh.bounding_box.centroid
    distances = np.linalg.norm(vertices, axis=1)
    vertices /= np.max(distances)
    if evaluate_metric:
        vertices /= 2
    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)


SIZE = None


class Render:
    def __init__(self, size, camera_pose, intensity=1.0, background=None):
        self.size = size
        global SIZE
        SIZE = size
        self.camera_pose = camera_pose
        self.background = background
        self.intensity = intensity

    def render(self, path, clean=True, mesh=None, only_render_images=False):
        mesh1 = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        rendered_image, depth = pyrender_rendering(
            mesh1,
            viz=False,
            light=True,
            camera_pose=self.camera_pose,
            intensity=self.intensity,
            bg_color=self.background,
        )

        return rendered_image, depth

    def render_pointclouds(self, mesh=None):
        mesh1 = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        pointclouds = pyrender_pointclouds(
            mesh1, camera_pose=self.camera_pose, bg_color=self.background
        )
        return pointclouds

    def render_normal(self, path, clean=True, mesh=None):
        try:
            if mesh.visual.defined:
                mesh.visual.material.kwargs["Ns"] = 1.0
        except:
            print("Error loading material!")

        triangle_id, depth_image, p_image = correct_normals(
            mesh, self.camera_pose, correct=True
        )

        return depth_image


def correct_normals(mesh, camera_pose, correct=True):
    rayintersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)

    a, b, index_tri, sign, p_image = trimesh_ray_tracing(
        mesh, camera_pose, resolution=SIZE * 2, rayintersector=rayintersector
    )
    if correct:
        mesh.faces[index_tri[sign > 0]] = np.fliplr(mesh.faces[index_tri[sign > 0]])

    normalmap = render_normal_map(
        pyrender.Mesh.from_trimesh(mesh, smooth=False),
        camera_pose,
        SIZE,
        viz=False,
    )

    return b, a, p_image


def init_light(scene, camera_pose, intensity=1.0):
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


class CustomShaderCache:

    def __init__(self):
        self.program = None

    def get_program(
        self, vertex_shader, fragment_shader, geometry_shader=None, defines=None
    ):
        if self.program is None:
            current_work_dir = os.path.dirname(__file__)
            print(current_work_dir)
            self.program = pyrender.shader_program.ShaderProgram(
                current_work_dir + "/shades/mesh.vert",
                current_work_dir + "/shades/mesh.frag",
                defines=defines,
            )
        return self.program


def render_normal_map(mesh, camera_pose, size, viz=False):
    scene = pyrender.Scene(bg_color=(255, 255, 255))
    scene.add(mesh)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    scene.add(camera, pose=camera_pose)

    renderer = pyrender.OffscreenRenderer(size, size)
    renderer._renderer._program_cache = CustomShaderCache()

    normals, depth = renderer.render(scene)

    world_space_normals = normals / 255 * 2 - 1

    if viz:
        image = Image.fromarray(normals, "RGB")
        image.show()

    return world_space_normals


def pyrender_rendering(
    mesh, camera_pose, viz=False, light=False, intensity=3.0, bg_color=None
):
    # renderer
    r = pyrender.OffscreenRenderer(SIZE, SIZE)

    scene = pyrender.Scene(bg_color=bg_color)
    scene.add(mesh)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 2.0, aspectRatio=1.0)
    camera = scene.add(camera, pose=camera_pose)
    # light
    if light:
        init_light(scene, camera_pose, intensity=intensity)

    scene.set_pose(camera, camera_pose)

    if light:
        color, depth = r.render(
            scene,
            flags=pyrender.constants.RenderFlags.ALL_SOLID
            | pyrender.constants.RenderFlags.FACE_NORMALS,
        )
    else:
        color, depth = r.render(scene, flags=pyrender.constants.RenderFlags.FLAT)

    return color, depth


def pyrender_pointclouds(mesh, camera_pose, bg_color=None):
    # renderer
    r = pyrender.OffscreenRenderer(SIZE, SIZE)

    scene = pyrender.Scene(bg_color=bg_color)
    scene.add(mesh)

    fov = np.pi / 2.0
    camera = pyrender.PerspectiveCamera(yfov=fov, aspectRatio=1.0)
    scene.add(camera, pose=camera_pose)

    _, depth = r.render(scene, flags=pyrender.constants.RenderFlags.FLAT)

    # Intrinsic matrix of the camera
    fy = fx = 0.5 / np.tan(fov * 0.5)  # assume aspectRatio is one.
    height = depth.shape[0]
    width = depth.shape[1]

    mask = np.where(depth > 0)

    x = mask[1]
    y = mask[0]

    normalized_x = (x.astype(np.float32) - width * 0.5) / width
    normalized_y = -1.0 * (y.astype(np.float32) - height * 0.5) / height

    world_x = normalized_x * depth[y, x] / fx
    world_y = normalized_y * depth[y, x] / fy
    world_z = -1.0 * depth[y, x]

    pointclouds = np.stack((world_x, world_y, world_z), axis=-1)  # [P, 3]
    pointclouds = (
        np.einsum("ij,kj->ki", camera_pose[:3, :3], pointclouds)
        + camera_pose[:3, 3][None, :]
    )  # [P, 3]

    return pointclouds  # [P, 3]


def trimesh_ray_tracing(mesh, M, resolution=225, fov=60, rayintersector=None):
    extra = np.eye(4)
    extra[0, 0] = 0
    extra[0, 1] = 1
    extra[1, 0] = -1
    extra[1, 1] = 0
    scene = mesh.scene()

    scene.camera_transform = M @ extra
    scene.camera.resolution = [resolution, resolution]
    scene.camera.fov = fov, fov
    origins, vectors, pixels = scene.camera_rays()

    index_tri, index_ray, points = rayintersector.intersects_id(
        origins, vectors, multiple_hits=False, return_locations=True
    )
    depth = trimesh.util.diagonal_dot(points - origins[0], vectors[index_ray])
    sign = trimesh.util.diagonal_dot(mesh.face_normals[index_tri], vectors[index_ray])

    pixel_ray = pixels[index_ray]
    a = np.zeros(scene.camera.resolution, dtype=np.uint8)
    b = np.ones(scene.camera.resolution, dtype=np.int32) * -1
    p_image = (
        np.ones(
            [scene.camera.resolution[0], scene.camera.resolution[1], 3],
            dtype=np.float32,
        )
        * -1
    )

    a[pixel_ray[:, 0], pixel_ray[:, 1]] = depth
    b[pixel_ray[:, 0], pixel_ray[:, 1]] = index_tri
    p_image[pixel_ray[:, 0], pixel_ray[:, 1]] = points

    return a, b, index_tri, sign, p_image
