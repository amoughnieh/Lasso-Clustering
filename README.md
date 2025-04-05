# Supervised Dimensionality Reduction for High-Cardinality Categorical Variables via Group Lasso Coefficient Clustering

## Overview

This project introduces and evaluates **Lasso Clustering (LC)**, a novel two-stage supervised methodology for dimensionality reduction of high-cardinality categorical variables. The approach utilizes **Group Lasso (GL)** to identify informative subcategories based on their regression coefficients, followed by **Mean Shift clustering** of these coefficients to create a compact feature representation. The effectiveness of LC is demonstrated on a subsample of the Rossmann Store Sales dataset, specifically targeting five key categorical variables (Store, DayOfWeek, StateHoliday, Month, Day), and benchmarked against the **Entity Embeddings (EE)** approach [1]. While achieving significant dimensionality reduction (>95%), the study explores the trade-offs between LC's interpretable clustering and the predictive performance of EE's dense embeddings.

## Baseline

The methodology and parts of the codebase are inspired by and adapted from the following work, which serves as the primary baseline for comparison:

* **Paper:** [1] Guo, C., & Berkhahn, F. (2016). *Entity Embeddings of Categorical Variables*. arXiv preprint arXiv:1604.06737. [https://doi.org/10.48550/arXiv.1604.06737](https://doi.org/10.48550/arXiv.1604.06737)
* **Repository:** [entron/entity-embedding-rossmann](https://github.com/entron/entity-embedding-rossmann)

## Data

### Source

The project utilizes data from the Kaggle competition: [Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales/overview).

### Required Data

Only the training data file, `train.csv`, from the competition is needed to run the primary analysis notebook `main.ipynb`.

**Important:** Due to its size, `train.csv` is **not** included in this repository. You must download it directly from the Kaggle competition page linked above.

### Included Files

To streamline the process and allow users to run the analysis without needing to execute the original preprocessing code from the baseline repository, the following precomputed files are included:

* `feature_train_data.pickle`: Contains the preprocessed training data, generated using code adapted from the baseline Entity Embeddings repository. Using this file bypasses the need to run the initial data preparation steps.
* `embeddings.pickle`: Contains the pre-trained entity embeddings. The notebook includes commented-out code for extracting them, if needed.

## Code Notes

The file `models_downstream.py` is largely based on the `models.py` file from the [entron/entity-embedding-rossmann](https://github.com/entron/entity-embedding-rossmann) repository. It has been slightly modified to:
1.  Incorporate a random seed for better reproducibility.
2.  Update certain sections for compatibility with newer versions of relevant libraries.

## Requirements

All required dependencies for running `main.ipynb` are listed in the included requirements file. Simply install them with:

`pip install -r requirements.txt`

## Getting Started

1.  Clone this repository.
2.  Download the `train.csv` file from the [Kaggle Rossmann Store Sales competition](https://www.kaggle.com/competitions/rossmann-store-sales/overview).
3.  Place the downloaded `train.csv` file into the root directory of this project.
4.  Ensure you have the necessary Python dependencies installed (see Requirements section above).
5.  Open and run the `main.ipynb` notebook.
---
