# Uncertainty-Aware Partial-Label Learning

This repository contains the code and text of the paper

> Anonymous Authors. "Uncertainty-Aware Partial-Label Learning"

submitted at the conference [ICML 2024](https://icml.cc/Conferences/2024).

This document provides (1) an outline of the repository structure and (2) steps to reproduce the experiments including setting up a virtual environment.

## Repository Structure

* The folder `data` contains all datasets used within our work.
  * The subfolder `realworld-datasets` contains commonly used real-world datasets for partial-label learning, which are initially provided by [Min-Ling Zhang](https://palm.seu.edu.cn/zhangml/Resources.htm).
  * The subfolder `ucipp` contains UCI datasets that have been used in our controlled experiments.
  The files are provided by Luis Paulo on [GitHub](https://github.com/lpfgarcia/ucipp).
* The folder `paper` contains all the plots that appear in the paper.
The paper's LaTeX source will be added once the camera-ready version is available.
* The folder `results` contains the results of all experiments as `.csv` files.
* The folder `src` contains the code of the experiments.
  * The subfolder `generate_plots` contains all python scripts to generate the plots in `paper/plots`.
  * The subfolder `partial_label_learning` contains all implementations of related work algorithms and our method.
  * `main_real_world.py` runs the experiments with the real-world datasets on one core.
  * `main.py` runs the experiments with the UCI datasets on one core.
  * `parallel_main_real_world.py` runs the experiments with the real-world datasets on all available cores.
  * `parallel_main.py` runs the experiments with the UCI datasets on all available cores.
* Additionally, there are the following files in the root directory:
  * `LICENSE` describes the repository's licensing.
  * `README.md` is this document.
  * `requirements.txt` is a list of all required `pip` packages for reproducibility.

## Setup

Before running scripts to reproduce the experiments, you need to set up an environment with all the necessary dependencies.
Our code is implemented in Python (version 3.11.5; other versions, including lower ones, might also work).

We used `virtualenv` (version 20.24.3; other versions might also work) to create an environment for our experiments.
First, you need to install the correct Python version yourself.
Next, you install `virtualenv` with

<table>
<tr>
<td> Linux + MacOS (bash-like) </td>
<td> Windows (powershell) </td>
</tr>
<tr>
<td>

``` sh
python -m pip install virtualenv==20.24.3
```

</td>
<td>

``` powershell
python -m pip install virtualenv==20.24.3
```

</td>
</tr>
</table>

To create a virtual environment for this project, you have to clone this repository first.
Thereafter, change the working directory to this repository's root folder.
Run the following commands to create the virtual environment and install all necessary dependencies:

<table>
<tr>
<td> Linux + MacOS (bash-like) </td>
<td> Windows (powershell) </td>
</tr>
<tr>
<td>

``` sh
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

</td>
<td>

``` powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

</td>
</tr>
</table>

## Reproducing the Experiments

Make sure that you created the virtual environment as stated above.
The script `parallel_main.py` runs the experiments on the UCI datasets on all available cores.
The script `parallel_main_real_world.py` runs the experiments on the real-world datasets on all available cores.
Running all experiments takes about a week on a system with 32 cores (Setup: `AMD EPYC 7551 32-Core Processor`).

<table>
<tr>
<td> Linux + MacOS (bash-like) </td>
<td> Windows (powershell) </td>
</tr>
<tr>
<td>

``` sh
python src/parallel_main.py
python src/parallel_main_real_world.py
```

</td>
<td>

``` powershell
python src\parallel_main.py
python src\parallel_main_real_world.py
```

</td>
</tr>
</table>

This creates `.csv` files in `results` containing the results of all experiments.

## Using the Data

The experiments' result files are plain CSVs.
You can easily read any of them with `pandas`.

``` python
import pandas as pd

datasets = pd.read_csv("results/datasets_used.csv")
```

## Generating Plots

To obtain plots from the data, use the python scripts in `src/generate_plots`.
Note that these scripts require a working installation of LaTeX on your local system.
Use the following snippets to generate all plots in the paper.
Generating all of them takes about 15 minutes on a single core.

<table>
<tr>
<td> Linux + MacOS (bash-like) </td>
<td> Windows (powershell) </td>
</tr>
<tr>
<td>

``` sh
python src/generate_plots/artificial_data_eval.py
python src/generate_plots/belief_simulation.py
python src/generate_plots/real_world_data_eval.py
```

</td>
<td>

``` powershell
python src\generate_plots\artificial_data_eval.py
python src\generate_plots\belief_simulation.py
python src\generate_plots\real_world_data_eval.py
```

</td>
</tr>
</table>
