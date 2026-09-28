# ENN555 – Operation and Maintenance of Renewable Energy Systems

Tutorial and helper code for ENN555 (QUT).

## Requirements

- Python 3.11 or newer
- Git
- A Gurobi licence for the dispatch/optimisation tutorials (see [Gurobi](#gurobi) below)

## Installation (editable)

The repo is installed as an *editable* package: Python imports the code directly from your clone, so any changes you make (or pull) take effect without reinstalling.

### 1. Clone the repository

```bash
git clone <repo-url> ENN555
cd ENN555
```

### 2. Create and activate a virtual environment

**venv (Windows, PowerShell)**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

**venv (macOS / Linux)**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

**conda (any platform)**

```bash
conda create -n enn555 python=3.11
conda activate enn555
```

### 3. Install the package

From the repo root (the folder containing `pyproject.toml`):

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

To also install Jupyter for the notebook tutorials:

```bash
python -m pip install -e ".[notebooks]"
```

Available extras:

| Extra       | Installs                 |
|-------------|--------------------------|
| `notebooks` | `jupyterlab`, `ipykernel` |
| `dev`       | `ruff`, `pytest`          |

### 4. Check the installation

```bash
python -c "from enn555 import paths; print(paths.repo_root()); print(paths.data_dir())"
```

This should print the path to your clone and its `data/` folder.

## Using the helpers

```python
import pandas as pd
import matplotlib.pyplot as plt
from enn555 import paths

df = pd.read_csv(paths.data_dir() / "brisbane_tmy.csv")
fig, ax = plt.subplots()
# ... plot ...
fig.savefig(paths.outputs_dir() / "my_figure.png")
```

- `paths.data_dir()` – the repo's `data/` folder (not tracked by git; place the course data files here).
- `paths.outputs_dir()` – the repo's `outputs/` folder, created on first use; a safe place for figures and results.

These functions locate the repo relative to the installed package, so they **only work with an editable install** (`pip install -e .`). A regular `pip install .` will not find `data/`.

## Gurobi

`gurobipy` is installed from PyPI automatically. The PyPI package includes a size-limited licence, which is enough for small models but not for the larger dispatch problems. Students and staff can get a free academic licence from Gurobi's [academic program](https://www.gurobi.com/academia/academic-program-and-licenses/); follow their instructions to activate it on your machine.

## Updating

```bash
git pull
```

No reinstall is needed after pulling, unless `pyproject.toml` has changed (e.g. new dependencies). In that case, re-run `python -m pip install -e .`.
