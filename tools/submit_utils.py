from pathos.multiprocessing import ProcessingPool as Pool
from rich.progress import track
import submitit
import time
import torch
import os
from typing import Optional, List, Dict, Callable, Any
import dataclasses
import functools
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from typing import Union

from tools.logger import get_logger


@dataclasses.dataclass
class Slurm:
    slurm_job_name: str = "training"  # Job name
    folder: str = "./log"  # Folder to save the log
    gpus_per_node: int = 0  # Number of GPUs per node
    nodes: int = 1  # Number of nodes
    slurm_constraint: str = (
        "[a100|rtx_a6000|rtx_3090|rtx_2080|gtx_1080]"  # '[a100|rtx_a6000|rtx_3090|rtx_2080|gtx_1080]'
    )
    cpus_per_task: int = 4  # Number of CPUs per task
    slurm_mail_type: Optional[str] = None  # "ALL"
    mem_gb: float = 40  # Memory per node
    slurm_time: str = "4-00:00:00"  # 4 days
    slurm_partition: str = "submit"
    stderr_to_stdout: bool = True  # Redirect stderr to stdout
    cluster: Optional[str] = None  # Use "local" to run jobs locally,
    timeout_min: int = int(1e10)  # Set to a large number to avoid timeout
    slurm_exclude: str = (
        "tarsonis,sorona,balrog,lothlann,moria,falas"  # "node1,node2,node3" uncomment for exclude nodes
    )
    # slurm_qos: str = "deadline"           # "deadline" uncomment for deadline queue
    chunk_size: int = 1


def group_dicts(list_dict: List[Dict[str, Any]], chunk_size: int = 1):
    """
    Group a list of dictionaries into chunks of a specified size.
    """
    # Initialize an empty list to hold the new dictionaries
    grouped_list = []

    # Iterate through the list in chunks
    for i in range(0, len(list_dict), chunk_size):
        chunk = list_dict[i : i + chunk_size]
        grouped_dict = defaultdict(
            list
        )  # Create a new dictionary where each value is a list

        # Populate the grouped dictionary
        for d in chunk:
            for key, value in d.items():
                grouped_dict[key].append(value)

        # Convert defaultdict to a normal dict and add to the final list
        grouped_list.append(dict(grouped_dict))

    return grouped_list


def slurm_func_wrapper(
    func: callable,
    num_workers: int = 1,
    fn_kwargs_share: Optional[dict] = {},
):
    """
    A wrapper function to call a function with a list of arguments.
    """

    @functools.wraps(func)
    def wrapper(fn_kwargs_list: Optional[list[str, dict]] = []):
        return run_with_mp(
            func,
            fn_kwargs_list=fn_kwargs_list,
            num_workers=num_workers,
            fn_kwargs_share=fn_kwargs_share.copy(),
        )

    return wrapper


def mp_func_wrapper(fn: Callable, fn_kwargs_share):
    """
    A wrapper function to call a function with a dictionary of keyword arguments.
    """

    @functools.wraps(fn)
    def wrapper(fn_kwargs: Union[Dict[str, Any], List[Any], Any]):
        if isinstance(fn_kwargs, dict):
            return fn(**fn_kwargs, **fn_kwargs_share.copy())
        elif isinstance(fn_kwargs, list):
            return fn(*fn_kwargs, **fn_kwargs_share.copy())
        else:
            return fn(fn_kwargs, **fn_kwargs_share.copy())

    return wrapper


def run_with_mp(
    fn: Callable,
    fn_kwargs_list: Optional[Union[List[Dict[str, Any]], List[Any]]] = [],
    num_workers: int = 1,
    fn_kwargs_share: Optional[Dict[str, Any]] = {},
):
    """
    A wrapper to run a function in parallel using multiprocessing with a pool of workers,
    with a progress bar.
    """
    # Create a logger within this function
    logger = get_logger(file_name=__file__)

    results = []
    if num_workers > 1:
        with Pool(processes=num_workers) as pool:
            fn_wrapped = mp_func_wrapper(fn, fn_kwargs_share=fn_kwargs_share)

            try:
                for result in pool.imap(fn_wrapped, fn_kwargs_list):
                    results.append(result)
            except Exception as e:
                logger.error(f"An error occurred: {e}")
                raise
    else:
        # Run the function sequentially if only one worker is specified
        for fn_kwargs in fn_kwargs_list:
            if isinstance(fn_kwargs, dict):
                result = fn(**fn_kwargs, **fn_kwargs_share)
            elif isinstance(fn_kwargs, list):
                result = fn(*fn_kwargs, **fn_kwargs_share)
            else:
                result = fn(fn_kwargs, **fn_kwargs_share)
            results.append(result)

    return results


def submit_jobs(
    fn: callable,
    fn_kwargs_list: Optional[list[str, dict]] = [],
    fn_kwargs_share: Optional[dict] = {},
    slurm_kwargs: dict = {},
    num_workers: int = 1,
    existing_jobs: List[Any] = None,
):
    """
    Submit a job to the cluster using the `submitit` library.

    Parameters:
    - fn: The function to run.
    - slurm_kwargs: A dictionary of SLURM parameters for the job.
    - fn_kwargs: Additional keyword arguments to pass to `fn`.
    - fn_kwargs_list: A list of parameters to distribute across the cluster. If None, the function will be run once.
    - wait_done: If True, wait for the job to finish before returning.
    - num_workers: The number of worker processes to use.

    Returns:
    - None
    """
    # Create a logger within this function
    logger = get_logger(file_name=__file__)

    # wait for old jobs to finish
    if existing_jobs is not None:
        num_finished = 0
        while num_finished < len(existing_jobs):
            if sum(job.done() for job in existing_jobs) > num_finished:
                num_finished = sum(job.done() for job in existing_jobs)
                logger.info(
                    f"Waiting for {len(existing_jobs) - num_finished} jobs to finish..."
                )
            time.sleep(10)

        results = []
        for i in range(len(existing_jobs)):
            results.extend(existing_jobs[i].result())  # TODO: check if this is correct

    if torch.cuda.is_available():
        logger.critical("CUDA is available, run the job locally")
        slurm_kwargs["cluster"] = "local"
    else:
        logger.critical("CUDA is not available, run the job on the cluster")

    folder = os.path.join(slurm_kwargs.pop("folder"), slurm_kwargs["slurm_job_name"])
    os.makedirs(folder, exist_ok=True)

    cluster = slurm_kwargs.pop("cluster")
    slurm_nodes = slurm_kwargs.pop("nodes")
    chunk_size = slurm_kwargs.pop("chunk_size", 1)

    if cluster == "local":
        executor = ThreadPoolExecutor(max_workers=num_workers)
    else:
        executor = submitit.AutoExecutor(folder=folder, cluster=cluster)
        executor.update_parameters(
            nodes=1, **slurm_kwargs
        )  # TUM VCG only supports 1 node each job

    if chunk_size > 1:
        assert (
            len(fn_kwargs_list) > 1
        ), "fn_kwargs_list must be provided when chunk_size > 1"
        fn_kwargs_list = group_dicts(fn_kwargs_list, chunk_size=chunk_size)

    jobs = []
    if len(fn_kwargs_list) == 0:
        jobs = [executor.submit(fn, **fn_kwargs_share)]
    else:
        num_jobs = len(fn_kwargs_list)
        chunk = num_jobs // min(num_jobs, slurm_nodes)

        func = slurm_func_wrapper(
            fn, num_workers=num_workers, fn_kwargs_share=fn_kwargs_share
        )

        for i in track(
            range(0, num_jobs, chunk), description="Submitting jobs to cluster"
        ):
            jobs.append(
                executor.submit(func, fn_kwargs_list=fn_kwargs_list[i : i + chunk])
            )

    logger.info(f"Submitted {len(jobs)} jobs")

    if cluster == "local":
        [job.result() for job in jobs]

    return jobs
