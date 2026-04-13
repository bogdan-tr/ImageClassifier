# **Image classification: Real vs Fake**

_Bogdan Trigubov_

This repository contains the source code for CNN model implementations for fake and real image detection. Please look at the PDF file in root for project details.

## General Directory Structure

```txt
.
├── condor
│   ├── jobs
│   │   ├── baseline_cnn_frequent_metrics
│   │   ├── baseline_cnn_lab4
│   │   ├── rs_baseline_fm
│   │   ├── rs_efficient_b0
│   │   ├── rs_efficient_b4
│   │   ├── rs_efficient_b4_pt
│   │   ├── rs_mesonet
│   │   ├── rs_mesonet_v2
│   │   ├── rs_mesonet_v3
│   │   ├── rs_vision_trans
│   │   ├── rs_xception
│   │   └── timm_model
│   ├── outputs
│   │   ├── baseline_cnn_frequent_metrics
│   │   ├── rs_baseline
│   │   ├── rs_efficient_b0
│   │   ├── rs_efficient_b4
│   │   ├── rs_efficient_b4_pt
│   │   ├── rs_mesonet
│   │   ├── rs_mesonet_v2
│   │   ├── rs_mesonet_v3
│   │   ├── rs_vision_trans
│   │   ├── rs_xception
│   │   └── timm_model
│   └── venv_package
├── experiments
│   ├── outputs
│   ├── output_txts
│   └── plotting
│   └── plots
├── main
│   ├── caller_bash_scripts
│   ├── models
│   ├── src
│   │   ├── helper_loads
│   │   │   └── **pycache**
│   │   └── **pycache**
│   └── test_script
├── notes
└── workspace
└── archive

```

> Note: Not all files/dirs are available on GitHub.

## `condor/`

`condor/` holds all the condor related contet. It is subdivided into `jobs/` and `outputs/` (not visible on GitHub). `jobs/` has a `job_log.txt` where are experiments were tracked and a separate folder for each model variation. Each model variation folder has the associated `.job` file and its wrapper bash script. Condor logs when to the `outputs/` directory.

## `experiments/`

`experiments/` contains all content related to hyperparameter tuning and plotting. Most folders in this directory are not on GitHub as they contain the _clutter_ of experimental data, tracking and plots. The `plotting/` subdirectory contains `plot.py` which generates loss/accuracy plots given a directory of csv log data. Usage: `python3 plot.py outputs` (where outputs is the directory of csv log data). The generated plots will all go to the `plots/` subdirectory.

## `main/`

`main/` contains all the logic and scripts for the directory.

### `caller_bash_scripts/`

This directory contains all the bash scripts called by condor for hyperparameter tuning and experimentation. Each bash script uses `>>` to put its printed output into a log file. Each file is called 10 times, but since random search is used in the scripts each run is different.

### `src/`

This directory contains the source code. All models and logic is found here.

- Each model can be run with `python3 <model>.py <run_name>`. `run_name` was used to keep track of the experiment.
- `FINAL_MODEL.py` is the model used for evaluation with the highest performance. It is a copy of `mesonet_imp_v3.py` with the hyperparameters fixed.
- `saliency.py` and `generate_saliency.sh` contain the code used to generate saliency maps. To run `saliency.py` use `python3 saliency.py <model pickle file> <output name> <model architecture abbreviation>`.
- `final_predict.py` contains the code used for the evaluation phase in the final predictions.

#### `helper_loads/`

This directory contains useful helper functions used by all the models.

- `load_dataloader.py` returns a dictionary with the validation and train dataloaders.
- `load_hyperparemeters.py` returns random hyperparameters within a certain range. Used for random search.
- `load_transfomrations` is a set of implemented, importable transformations for the dataloader.

## `notes/`

This directory contains `observations.md` which contains the patterns I found by manually inspecting the data and `process.md` which I used to keep track of progress and tasks.
