from mpi4py import MPI
from typing import Any
from tqdm import tqdm
import json
import logging
import time
import polars as pl
from pathlib import Path
from util.task import Task, TaskType
from model.data import DataModel
from model.dataset import generate_data
from erm.optimize import start_optimization
from sweep.results.erm import ERMResult
from sweep.results.state_evolution import SEResult
from state_evolution.iteration import fixed_point_finder
from sweep.experiment import NumpyEncoder


def run_erm(task: Task, data_model: DataModel) -> ERMResult:
    """
    Generate Data, run ERM and return ERMResult
    """
    logging.info(f"Starting ERM {task}")
    start = time.time()

    data = generate_data(
        data_model, n=int(task.alpha * task.d), tau=task.tau, seed=task.id
    )

    weights_erm, values = start_optimization(task, data_model, data)

    erm_result = ERMResult(task, data_model, data, weights_erm, values)

    end = time.time()
    erm_result.duration = end - start

    logging.info(f"Finished ERM {task}")

    return erm_result


def run_state_evolution(task, data_model) -> SEResult:
    """
    Starts the state evolution and returns SEResult
    """

    logging.info(f"Starting State Evolution {task}")
    start = time.time()

    overlaps = fixed_point_finder(data_model, task)

    st_exp_info = SEResult(task, overlaps, data_model)

    end = time.time()
    experiment_duration = end - start
    st_exp_info.duration = experiment_duration

    logging.info(f"Finished State Evolution {task}")

    return st_exp_info


# Define a function to process a task
def process_task(task) -> Any:
    try:
        logging.info(f"Starting task {task.id}")

        # get the data model
        data_model: DataModel = task.data_model
        logging.debug(data_model)
        data_model.generate_model_matrices()

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


# Define the master function
def master(num_processes, experiment) -> None:
    logging.info("Starting Experiment %s", experiment.name)

    # note starttime
    start = time.time()

    tasks = list(experiment)

    # Initialize the progress bar
    pbar = tqdm(total=len(tasks))

    # start the processes
    logging.info("Starting all processes")
    # Send the tasks to the workers
    task_idx = 0
    received_tasks = 0
    for i in range(num_processes):
        if task_idx >= len(tasks):
            break
        task = tasks[task_idx]
        logging.info(f"Sending task {task_idx} to {i+1}")
        MPI.COMM_WORLD.send(task, dest=i + 1, tag=task.id)
        task_idx += 1

    erm_results = []
    se_results = []

    logging.info("All processes started - receiving results and sending new tasks")
    # Receive and store the results from the workers
    while received_tasks < len(tasks):
        status = MPI.Status()
        # log status information
        logging.info(f"Received the {received_tasks}th task")

        task = MPI.COMM_WORLD.recv(
            source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status
        )
        received_tasks += 1

        logging.info(f"Received task {task.id} from {status.source}")

        # get the result
        result = task.result
        logging.debug(result)

        # test if the result is an exception
        if not isinstance(result, Exception):
            match task.task_type:
                case TaskType.ERM:
                    erm_results.append(vars(result))
                case TaskType.SE:
                    se_results.append(vars(result))

            logging.info(f"Saved {task}")
        else:
            logging.error(f"Error {task}")
            logging.error(result)

        # Update the progress bar
        pbar.update(1)

        # Send the next task to the worker that just finished
        if task_idx < len(tasks):
            task = tasks[task_idx]
            MPI.COMM_WORLD.send(task, dest=status.source, tag=task.id)
            task_idx += 1

    logging.info("All tasks sent and received")

    directory = Path("results") / experiment.name
    directory.mkdir(parents=True, exist_ok=True)

    # TODO, eventually just saving the dataframe should be enough...
    with open(directory / "erm_results.json", "w") as f:
        f.write(json.dumps(erm_results, cls=NumpyEncoder))

    with open(directory / "se_results.json", "w") as f:
        f.write(json.dumps(se_results, cls=NumpyEncoder))

    # Leverage the existing NumpyEncoder to create a polars dataframe without "object" type
    df_erm = pl.DataFrame(json.loads(json.dumps(erm_results, cls=NumpyEncoder)))
    df_se = pl.DataFrame(json.loads(json.dumps(se_results, cls=NumpyEncoder)))

    with open(directory / "df_erm.ser", "wb") as f:
        f.write(df_erm.serialize())

    with open(directory / "df_se.ser", "wb") as f:
        f.write(df_se.serialize())

    # mark the experiment as finished
    logging.info(f"Marking experiment {experiment.name} as finished")
    end = time.time()
    duration = end - start
    logging.info("Experiment took %d seconds", duration)

    # Close the progress bar
    pbar.close()

    logging.info("All done - signaling exit")
    # signal all workers to stop
    for i in range(num_processes):
        MPI.COMM_WORLD.send(None, dest=i + 1, tag=0)
