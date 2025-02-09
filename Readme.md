# A High Dimensional Model for Adversarial Training: Geometry and Trade-Offs


To reproduce the figures from the paper, please use `define_experiment.ipynb` in the experiments folder.

All experiments have a definition for the data-model and problem types considered. The choice of the sweep parameters has to be customised.

Once the experiment has been defined, and a json file containing the data-model and sweep definition (usually called `sweep_experiment.json`) been created, the data-models have to be created using the `create_data_model.py` script.
Then, the experiment can be exectued using `sweep.py`.
Running these scripts requires a working MPI installation. The command sequence is
```bash
mpiexec -n 5 python create_data_model.py sweep_experiment.json
mpiexec -n 5 python sweep.py sweep_experiment.json
```
Alternatively, in a cluster environment, it is possible to use the generated `run.sh` file.

The `sweep.py` script stores all results in a sqlite database. We provide scripts to easily extract the data in the `Evaluate` folder.

The experiments on real data have been performed in the `pca_experiments.ipynb` notebook.


## How to use

Install mpi using brew
```bash
brew install mpich
```

Install an environment for execution:
```bash
pip install uv
uv venv --python 3.11

source .venv/bin/activate
uv pip install -r pyproject.toml
```

## How to contibute

Make the bump-version script executable:
```bash
chmod +x hooks/bump-version.sh
```

Install mpi using brew
```bash
brew install mpich
```

Install an environment for development:
```bash
pip install uv
uv venv --python 3.11

source .venv/bin/activate
uv pip install -r pyproject.toml --extra dev
```

Install a pre-commit hook for ruff
```bash
pre-commit install
```


Run the tests like this to see logging output
```bash
pytest -o log_cli=true -o log_cli_level=INFO
```
