[Chinese Documentation](README.zh-CN.md)
# Project Name

## Citation

## Installation

A `requirements.txt` file is provided with details of all packages needed to recreate the virtual environment used for implementation and code execution. Install the dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Using the Code and Analysis Files

### Random Sample Generator (`value_combination.py`)

See the instructions in the code for detailed usage. Before running the Python program, adjust the hyperparameters to match the desired format.

The program generates a specified number of random samples based on the configured features, their value ranges, and sampling methods.

### Active Learning Program (`tib_active.py`)

The core program uses portions of the METIS code released as open source by [^1].

[^1]:	PANDI A，DIEHL C，YAZDIZADEH KHARRAZI A，et al. A versatile active learning workflow for optimization of genetic and metabolic networks[J/OL]. Nature Communications，2022，13（1）：3876. [DOI:10.1038/s41467-022-31245-z](https://doi.org/10.1038/s41467-022-31245-z)



To use the program, modify the corresponding parameters near the bottom of the file.

Specify the paths to the training dataset and candidate sample dataset to obtain an ensemble of the 20 best selected models and upper confidence bound (UCB) scores for all candidate samples.

### SHAP Analysis (`SHAP_titer.ipynb`, `SHAP_cost.ipynb`)

See the notebooks for detailed usage instructions. They include additional analyses that were not all used in this paper.

The notebooks can train a model on existing data or use an existing model to perform basic model performance evaluation, SHAP main effect analysis, and SHAP interaction effect analysis.

### Model Analysis (`model_analysis.ipynb`)

See the notebook for detailed usage instructions.

The program trains models on existing data and reports $R^2$, $Spearman$, and $MAE$ for each round under three evaluation settings: training and test sets drawn from the same batch of data through five-fold cross-validation; model-recommended formulations used as the test set; and the final round of data used as a common test set.

## Complete Workflow

### Active Learning Loop

After collecting the initial data, use the random sample generator (`value_combination.py`) to generate a candidate sample dataset. Then use the active learning program (`tib_active.py`) to obtain trained models and UCB scores for the candidate samples. Select high-scoring candidates for experiments in the next iteration.

### Analysis After Completing the Experiments

Combine all experimental results into a single data table, such as `ALL_exp._results.csv`, then use that table for SHAP analysis and model analysis (`model_analysis.ipynb`).



