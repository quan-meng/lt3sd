import os
import numpy as np
import matplotlib.pyplot as plt

from tools.submit_utils import run_with_mp
from tools.logger import get_logger

logger = get_logger(file_name=__file__, debug="get_stats")


def get_mean_min_max(file_path):
    data = np.load(file_path, allow_pickle=True)  # [C, G1, G2, G3]

    return {
        "mean_val": np.mean(data, axis=(1, 2, 3)),  # [C]
        "min_val": np.min(data, axis=(1, 2, 3)),  # [C]
        "max_val": np.max(data, axis=(1, 2, 3)),  # [C]
    }


def get_var(file_path, mean_val):
    data = np.load(file_path, allow_pickle=True)  # [C, G1, G2, G3]
    var = np.mean((data - mean_val[:, None, None, None]) ** 2, axis=(1, 2, 3))  # [C]

    return var


def check_outliers(file_path, lower_bound, upper_bound):
    lower_bound = lower_bound[..., None, None, None]
    upper_bound = upper_bound[..., None, None, None]

    data = np.load(file_path, allow_pickle=True)  # [C, G1, G2, G3]
    num_outliers = np.sum((data < lower_bound) + (data > upper_bound)) / data.size

    return np.array(num_outliers)


def get_hist_counts(file_path, num_bins=1000, bounds=None):
    data = np.load(file_path, allow_pickle=True).flatten()
    hist, bin_edges = np.histogram(data, bins=num_bins, range=(bounds[0], bounds[1]))
    return hist, bin_edges


def get_stats(file_paths, stats_dir, stds=5, num_workers=16):
    num_shapes = len(file_paths)

    # Get the mean, min, max -----------------------------------------------------------------
    fn_kwargs_list = [{"file_path": x} for x in file_paths]
    results = run_with_mp(get_mean_min_max, fn_kwargs_list, num_workers=num_workers)

    statistics = {
        "mean_val": [],
        "min_val": [],
        "max_val": [],
        "std": 0.0,
        "lower_bound": 0.0,
        "upper_bound": 0.0,
    }
    for i in range(len(results)):
        statistics["mean_val"].append(results[i]["mean_val"])
        statistics["min_val"].append(results[i]["min_val"])
        statistics["max_val"].append(results[i]["max_val"])
    statistics["mean_val"] = np.mean(statistics["mean_val"], axis=0)  # [C]
    statistics["min_val"] = np.min(statistics["min_val"], axis=0)  # [C]
    statistics["max_val"] = np.max(statistics["max_val"], axis=0)  # [C]

    # Get the std, lower_bound, upper_bound -----------------------------f-----------------------
    mean_val = statistics["mean_val"]
    fn_kwargs_list = [{"file_path": x, "mean_val": mean_val} for x in file_paths]
    results = run_with_mp(get_var, fn_kwargs_list, num_workers=num_workers)

    std = np.sqrt(sum(results) / num_shapes)  # [C]
    statistics["std"] = std
    statistics["lower_bound"] = mean_val - (stds * std)
    statistics["upper_bound"] = mean_val + (stds * std)
    logger.info(f"Statistics: {statistics}")

    # Save stats -----------------------------------------------------------------
    npz_path = os.path.join(stats_dir, "statistics.npz")
    np.savez(npz_path, **statistics)
    logger.info(f"Saved statistics to {npz_path}")

    # Check outliers -----------------------------------------------------------------
    lower_bound = statistics["lower_bound"]
    upper_bound = statistics["upper_bound"]

    fn_kwargs_list = [{"file_path": x} for x in file_paths]
    fn_kwargs_share = {"lower_bound": lower_bound, "upper_bound": upper_bound}
    results = run_with_mp(
        check_outliers,
        fn_kwargs_list,
        num_workers=num_workers,
        fn_kwargs_share=fn_kwargs_share,
    )

    num_outliers = 0.0
    for result in results:
        num_outliers += result
    logger.info(f"Outliers: {num_outliers / num_shapes * 100}%")

    # Plot histogram -----------------------------------------------------------------
    num_bins = 1000
    fn_kwargs_list = [{"file_path": x} for x in file_paths]
    fn_kwargs_share = {
        "num_bins": num_bins,
        "bounds": (min(statistics["min_val"]), max(statistics["max_val"])),
    }
    results = run_with_mp(
        get_hist_counts,
        fn_kwargs_list,
        num_workers=num_workers,
        fn_kwargs_share=fn_kwargs_share,
    )

    hist_counts = np.zeros(num_bins)
    for result in results:
        hist_counts += result[0]
        bin_edges = result[1]

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(bin_edges[:-1], hist_counts, width=np.diff(bin_edges))
    plt.xlabel("Bins")
    plt.ylabel("Count")
    plt.title("Histogram of Dataset")
    plt.savefig(os.path.join(stats_dir, f"stats.png"))
