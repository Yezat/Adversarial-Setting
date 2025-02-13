# usage: mpiexec -n 5 python run_sweep.py --file experiment.json

import logging
import sys
import argparse
import json
from sweep.experiment import Experiment, NumpyDecoder
from mpi4py import MPI
from helpers import master, worker


def setup_logging(log_level: int, rank: int) -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # StreamHandler for the root process
    if rank == 0:
        handler = logging.StreamHandler(sys.stdout)
    else:
        handler = logging.FileHandler(
            f"log/log_rank_{rank}.log", mode="w"
        )  # separate log for each non-root process

    handler.setLevel(log_level)

    formatter = logging.Formatter(
        f"%(asctime)s - Rank {rank} - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Accept either a filename to experiment.json or a JSON string."
    )

    parser.add_argument(
        "--file", type=str, help="Path to a JSON file with a serialised experiment."
    )
    parser.add_argument(
        "--json", type=str, help="Serialised JSON experiment string input."
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",  # Default log level
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],  # Valid options
        help="Set the logging level",
    )

    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)

    # Ensure that at least one argument is provided
    if not args.file and not args.json:
        parser.error("At least one of --file or --json must be provided.")

    if args.file:
        try:
            with open(args.file, "r") as f:
                data = json.load(f, cls=NumpyDecoder)
        except Exception as e:
            parser.error(f"Failed to read file: {e}")

    if args.json:
        try:
            data = json.loads(args.json, cls=NumpyDecoder)
        except json.JSONDecodeError as e:
            parser.error(f"Invalid JSON string: {e}")

    experiment = Experiment.fromdict(data)

    # create the MPI communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    logger = setup_logging(log_level, rank)

    logger.info("The MPI comm size is %d", size)

    logger.info("This process has rank %d", rank)

    if rank == 0:
        # run the master
        logger.info("Starting Master for experiment %s", experiment.name)
        master(size - 1, experiment)

    else:
        # run the worker
        worker()

    MPI.Finalize()
