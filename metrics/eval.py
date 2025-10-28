import os
import trimesh
import torch
import json
import yaml
import pathlib
import random
import StructuralLosses as stb
from pytorch_fid.fid_score import (
    InceptionV3,
    IMAGE_EXTENSIONS,
    calculate_activation_statistics,
    calculate_frechet_distance,
)


from tools.submit_utils import run_with_mp


def compute_statistics_of_path(
    path, model, batch_size, dims, device, num_workers=1, n_sample=-1
):
    if isinstance(path, str):
        path = [path]

    files = []
    for p in path:
        if not os.path.exists(p):
            raise RuntimeError("Invalid path: %s" % p)

        p = pathlib.Path(p)
        files_i = sorted(
            [file for ext in IMAGE_EXTENSIONS for file in p.glob("*.{}".format(ext))]
        )
        files += files_i

    if n_sample > 0:
        random.shuffle(files)
        files = files[:n_sample]

    m, s = calculate_activation_statistics(
        files, model, batch_size, dims, device, num_workers
    )

    return m, s


def calculate_fid_given_paths(
    paths, batch_size, device, dims, num_workers=1, n_sample=-1
):
    """Calculates the FID of two paths"""
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    m1, s1 = compute_statistics_of_path(
        paths[0], model, batch_size, dims, device, num_workers, n_sample=n_sample
    )
    m2, s2 = compute_statistics_of_path(
        paths[1], model, batch_size, dims, device, num_workers
    )
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def load_pclouds(path):
    pclouds = torch.tensor(trimesh.load(path).vertices).float()  # [N, 3]
    nponts = len(pclouds)

    mask = pclouds[..., 1] < 0.8 # mask the ceiling
    pclouds = pclouds[mask]

    if len(pclouds) == 0:
        return None
    else:
        pclouds = pclouds[torch.arange(nponts) % len(pclouds)]

    return pclouds


def calculate_cd_cmd(pd_paths, gt_paths, num_workers=1, batch_size=8):
    pd_paths = [{"path": p} for p in pd_paths]
    gt_paths = [{"path": p} for p in gt_paths]

    pd_pclouds = run_with_mp(
        load_pclouds, fn_kwargs_list=pd_paths, num_workers=num_workers
    )
    gt_pclouds = run_with_mp(
        load_pclouds, fn_kwargs_list=gt_paths, num_workers=num_workers
    )

    pd_pclouds = [p for p in pd_pclouds if p is not None]
    gt_pclouds = [p for p in gt_pclouds if p is not None]

    print(f"pd_size: {len(pd_pclouds)}, gt_size: {len(gt_pclouds)}")

    pd_pclouds = torch.stack(pd_pclouds).to("cuda:0")  # [B, N, 3]
    gt_pclouds = torch.stack(gt_pclouds).to("cuda:0")  # [B, N, 3]

    results = stb.compute_all_metrics(pd_pclouds, gt_pclouds, batch_size=batch_size)

    print(f"CD&EMD-Metrics: {results}")

    return results


def eval_fid_2d(opt=None):
    # read yaml file as a dict
    with open(opt.yaml_path, "r") as f:
        config = yaml.safe_load(f)[opt.category]

    pd_dirs = config[opt.model]["images"]
    gt_dirs = config[opt.task.gt]["images"]

    pd_dirs = [pd_dirs] if isinstance(pd_dirs, str) else pd_dirs
    gt_dirs = [gt_dirs] if isinstance(gt_dirs, str) else gt_dirs

    fid = calculate_fid_given_paths(
        [pd_dirs, gt_dirs],
        batch_size=opt.task.batch_size,
        device="cuda:0",
        dims=2048,
        num_workers=8,
        n_sample=opt.task.n_sample,
    )

    # write the dict to a yaml file
    prefix = "_".join(opt.category)
    out_path = os.path.join(f"./log/{opt.model}_{prefix}_fid.txt")
    results = {"FID-2D": fid}
    print(f"FID-2D: {fid}")

    # export the results (dict{}) with preety format to a txt file of path out_path with indent
    with open(out_path, "w") as file:
        file.write(json.dumps(results, indent=4))
