#!/bin/bash
#SBATCH --chdir=/home/ktanner/TPIV---Logistic-Classification/experiments
#SBATCH --job-name=TPIV-Adversarial
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=1
#SBATCH --mem=200G
#SBATCH --output='./out.txt'
#SBATCH --error='./error.txt'
#SBATCH --time=24:00:00

module purge
module load gcc openmpi python/3.11.0
source /home/ktanner/venvs/adv/bin/activate

srun --mpi=pmi2 python3 ./sweep.py --file sweep_experiment.json

deactivate
