import os
import trimesh
import numpy as np
import torch
import h5py
import tyro
import sys
import dataclasses
import glob
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from configs.dataset import Front3D
from tools.submit_utils import submit_jobs, Slurm
from tools.logger import get_logger

logger = get_logger(file_name=__file__, debug="export_volume")

sdf_gen_cmd = (
    lambda in_filepath, out_filepath, grid_size, padding: f"./tools/sdf_gen {in_filepath} {out_filepath} {grid_size} {padding}"
)


def export_tudf_volume(
    mesh_path: str,
    out_path: str,
    voxel_size: float = 0.022,
    num_level: int = 4,
    padding: int = 1,
    with_bbox: bool = False,
    skip_sdfgen: bool = False,
):
    if not skip_sdfgen:
        os.system(sdf_gen_cmd(mesh_path, out_path, voxel_size, padding))

        if os.path.exists(out_path + "_if.npy"):
            os.remove(out_path + "_if.npy")

    scene_name = os.path.basename(mesh_path).replace(".obj", "")
    dir_path = os.path.dirname(os.path.dirname(out_path))
    if os.path.exists(out_path + ".npy"):
        x = np.load(out_path + ".npy")
        x = torch.from_numpy(x).float()[None, None]  # [1, 1, G, G, G]
        for k in range(1, num_level):
            vxl_size_i = voxel_size * (2**k)
            x = F.avg_pool3d(x, kernel_size=2)  # [B, 1, g, g, g]
            out_path_k = os.path.join(
                dir_path, f"udf_voxel_{vxl_size_i}", f"{scene_name}.npy"
            )
            np.save(out_path_k, x[0, 0].numpy())
    else:
        logger.warning(f"No {out_path + '.npy'} found, skip {scene_name}")

    if with_bbox:
        mesh = trimesh.load(mesh_path)

        with h5py.File(mesh_path.replace(".obj", "_bbox.h5"), "r") as h5file:
            bbox = h5file["bbox"][:]
            category = [s.decode() for s in h5file["category"][:]]

        scene_min = np.min(mesh.vertices, axis=0) - padding * voxel_size  # [3]
        scene_max = np.max(mesh.vertices, axis=0) + padding * voxel_size  # [3]

        # [N, 2, 3] in [0, 1.0]
        bbox = (bbox - scene_min[None, None]) / (scene_max - scene_min)[None, None]

        with h5py.File(
            os.path.join(dir_path, "bbox", f"{scene_name}.h5"), "w"
        ) as h5file:
            h5file.create_dataset("bbox", data=bbox)
            h5file.create_dataset("category", data=np.string_(category))


def main(
    slurm: Slurm,
    voxel_size: float = 0.022,
    num_level: int = 4,
    category: str = "House",
    with_bbox: bool = True,
    num_workers: int = 1,
    skip_sdfgen: bool = False,
    padding: int = 1,
):
    for l in range(num_level):
        vxl_size_i = voxel_size * (2**l)
        out_dir_k = os.path.join(Front3D.root_dir, category, f"udf_voxel_{vxl_size_i}")
        os.makedirs(out_dir_k, exist_ok=True)

    os.makedirs(os.path.join(Front3D.root_dir, category, "bbox"), exist_ok=True)

    fn_kwargs_share = {
        "padding": padding,
        "with_bbox": with_bbox,
        "skip_sdfgen": skip_sdfgen,
    }

    mesh_paths = sorted(
        glob.glob(os.path.join(Front3D.root_dir, "3D-FRONT-house", f"*.obj"))
    )
    out_paths = [
        os.path.join(
            Front3D.root_dir,
            category,
            f"udf_voxel_{voxel_size}",
            os.path.basename(path).strip(".obj"),
        )
        for path in mesh_paths
    ]
    fn_kwargs_list = [
        {"mesh_path": mesh_path, "out_path": out_path}
        for mesh_path, out_path in zip(mesh_paths, out_paths)
    ]

    logger.info(f"Processing {len(fn_kwargs_list)} files to {Front3D.root_dir}")

    submit_jobs(
        fn=export_tudf_volume,
        fn_kwargs_list=fn_kwargs_list,
        fn_kwargs_share=fn_kwargs_share,
        slurm_kwargs=dataclasses.asdict(slurm),
        num_workers=num_workers,
    )


if __name__ == "__main__":
    tyro.cli(main)
