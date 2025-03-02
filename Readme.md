# A High Dimensional Model for Adversarial Training: Geometry and Trade-Offs


To reproduce the figures from the paper, please use `define_experiment.ipynb` in the experiments folder.

All experiments have a definition for the data-model and problem types considered. The choice of the sweep parameters has to be customised.

Once you've created an `experiment.json` file and you would like to run the experiment from the command line instead of the jupyter notebook, use:
```bash

mpiexec -n 5 python sweep/run_sweep.py --file experiment.json
```
Alternatively, in a cluster environment, it is possible to use a `run.sh` file. There's an example in experiments/run.sh.

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

Install this package
```bash
uv pip install .
```

Build brentq.c:
```bash
gcc -shared -o numerics/brentq.so numerics/brentq.c
```

## How to contibute

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

Install this package in editable mode
```bash
uv pip install -e .
```

Build brentq.c:
```bash
gcc -shared -o numerics/brentq.so numerics/brentq.c
```

Install a pre-commit hook for ruff
```bash
pre-commit install
```


Run the tests like this to see logging output
```bash
pytest -o log_cli=true -o log_cli_level=INFO
```


If you work with vs code, add this to your .vscode/settings.json file in your workspace for the notebooks to resolve your python path correctly:
```json
"python.analysis.extraPaths": [
        "${workspaceFolder}"
    ]
```