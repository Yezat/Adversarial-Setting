# usage: mpiexec -n 5 python sweep.py sweep_experiment.json

from mpi4py import MPI
from typing import Any
from tqdm import tqdm
import logging
import time
import sys
from sweep.experiment import Experiment
from util.task import Task, TaskType
from util.serialisation import NumpyDecoder
from model.data import DataModel
from erm.optimize import start_optimization
from sweep.results.erm import ERMResult
from sweep.results.state_evolution import SEResult
from state_evolution.iteration import fixed_point_finder
import json


def run_erm(task: Task, data_model: DataModel) -> ERMResult:
    """
    Generate Data, run ERM and save the results to the database
    """
    logging.info(f"Starting ERM {task}")
    start = time.time()

    data = data_model.generate_data(int(task.alpha * task.d), task.tau)

    weights_erm, values = start_optimization(task, data_model, data)

    erm_results = ERMResult(task, data_model, data, weights_erm, values)

    end = time.time()
    erm_results.duration = end - start

    logging.info(f"Finished ERM {task}")

    return erm_results


def run_state_evolution(task, data_model) -> SEResult:
    """
    Starts the state evolution and saves the results to the database
    """

    logging.info(f"Starting State Evolution {task}")
    start = time.time()

    overlaps = fixed_point_finder(data_model, task)

    st_exp_info = SEResult(task, overlaps, data_model)

    end = time.time()
    experiment_duration = end - start
    st_exp_info.duration = experiment_duration

    logging.info(f"Finished State Evolution {task}")
    overlaps.log_overlaps(logging)

    return st_exp_info


# Define a function to process a task
def process_task(task, data_model) -> Any:
    try:
        logging.info(f"Starting task {task.id}")

        # get the data model
        data_model = eval(task.data_model)

        match task.task_type:
            case TaskType.SE:
                task.result = run_state_evolution(task, data_model)
            case TaskType.ERM:
                task.result = run_erm(task, data_model)

    except Exception as e:
        # log the exception
        logging.exception(e)
        # set the result to the exception
        task.result = e

    return task


# Define the worker function
def worker() -> None:
    # get the rank
    rank = MPI.COMM_WORLD.Get_rank()

    while True:
        try:
            # Receive a task from the master
            task = MPI.COMM_WORLD.recv(source=0, tag=MPI.ANY_TAG)
            if task is None:
                # Signal to exit if there are no more tasks
                logging.info(f"Received exit signal - my rank is {rank}")
                break

            # Process the task
            result = process_task(task)
            # Send the result to the master
            MPI.COMM_WORLD.send(result, dest=0, tag=task.id)
        except Exception as e:
            # log the exception
            logging.exception(e)
            MPI.COMM_WORLD.send(e, dest=0, tag=0)
            return

    logging.info(f"Worker exiting - my rank is {rank}")


def load_experiment(filename) -> Any | None:
    # Get the experiment information from this file.
    if filename is None:
        filename = "sweep_experiment.json"

    # load the experiment parameters from the json file
    try:
        with open(filename) as f:
            experiment_dict = json.load(f, cls=NumpyDecoder)
            experiment = Experiment.fromdict(experiment_dict)
            logging.info("Loaded experiment from file %s", filename)

            return experiment
    except FileNotFoundError:
        logging.error(
            "Could not find file %s. Using the standard elements instead", filename
        )


# Define the master function
def master(num_processes, logger, experiment) -> None:
    experiment_id = experiment.experiment_id
    logger.info("Starting Experiment with id %s", experiment_id)

    # note starttime
    start = time.time()

    tasks = list(experiment)

    # Initialize the progress bar
    pbar = tqdm(total=len(tasks))

    # start the processes
    logger.info("Starting all processes")
    # Send the tasks to the workers
    task_idx = 0
    received_tasks = 0
    for i in range(num_processes):
        if task_idx >= len(tasks):
            break
        task = tasks[task_idx]
        logger.info(f"Sending task {task_idx} to {i+1}")
        MPI.COMM_WORLD.send(task, dest=i + 1, tag=task.id)
        task_idx += 1

    logger.info("All processes started - receiving results and sending new tasks")
    # Receive and store the results from the workers
    while received_tasks < len(tasks):
        status = MPI.Status()
        # log status information
        logger.info(f"Received the {received_tasks}th task")

        task = MPI.COMM_WORLD.recv(
            source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status
        )
        received_tasks += 1

        logger.info(f"Received task {task.id} from {status.source}")

        # result
        result = task.result

        # test if the result is an exception
        if not isinstance(result, Exception):
            match task.task_type:
                case TaskType.ERM:
                    raise NotImplementedError
                case TaskType.SE:
                    raise NotImplementedError

            logger.info(f"Saved {task}")
        else:
            logger.error(f"Error {task}")

        # Update the progress bar
        pbar.update(1)
        logger.info("")
        # Send the next task to the worker that just finished
        if task_idx < len(tasks):
            task = tasks[task_idx]
            MPI.COMM_WORLD.send(task, dest=status.source, tag=task.id)
            task_idx += 1

    logger.info("All tasks sent and received")

    # mark the experiment as finished
    logger.info(f"Marking experiment {experiment_id} as finished")
    end = time.time()
    duration = end - start
    logger.info("Experiment took %d seconds", duration)

    # Close the progress bar
    pbar.close()

    logger.info("All done - signaling exit")
    # signal all workers to stop
    for i in range(num_processes):
        MPI.COMM_WORLD.send(None, dest=i + 1, tag=0)


if __name__ == "__main__":
    # create the MPI communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # read the filename from the command line
    filename = sys.argv[1]

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        f"%(asctime)s - %(levelname)s - rank {rank} - %(message)s"
    )
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info("The MPI comm size is %d", size)

    logger.info("This process has rank %d", rank)

    experiment = load_experiment(filename, logger)

    if rank == 0:
        # run the master
        master(size - 1, experiment)

    else:
        # run the worker
        worker(logger, experiment)

    MPI.Finalize()
