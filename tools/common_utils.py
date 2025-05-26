import torch
import copy
import importlib
from inspect import isfunction
from tools.logger import get_logger

logger = get_logger(file_name=__file__, debug="common_utils")


def parse_level(level):
    data_type, vxl_size_i, vxl_size_o = level.replace("p", ".").split("_")
    return data_type, float(vxl_size_i), float(vxl_size_o)


def ada_threshold(vxl_size, mode, factor=1.5):
    assert mode in ["tsdf", "tudf"], f"mode: {mode} is not supported!"
    return 0.0 if mode == "tsdf" else vxl_size * factor


def int2tuple(x, dims=3):
    return x if isinstance(x, (list, tuple)) else [x] * dims


def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(cfg_dict):
    cfg_dict = copy.deepcopy(cfg_dict)
    target = cfg_dict.pop("target")
    if "name" in cfg_dict:
        cfg_dict.pop("name")

    return get_obj_from_str(target)(**cfg_dict)


def recursive_to(a, device):
    if isinstance(a, dict):
        return {k: recursive_to(v, device) for k, v in a.items()}
    elif isinstance(a, torch.Tensor):
        return a.to(device)
    elif isinstance(a, list):
        return [recursive_to(v, device) for v in a]
    elif isinstance(a, int) or isinstance(a, float) or isinstance(a, str) or a is None:
        return a
    else:
        raise NotImplementedError


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def get_model_size(model):
    num_params = count_parameters(model)
    model_size_mb = (
        num_params * 4 / (1024**2)
    )  # Convert bytes to megabytes (1 float32 = 4 bytes)

    return f"{model_size_mb:.2f} MB"
