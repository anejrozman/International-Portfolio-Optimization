# Multi-Currency Portfolio Optimization with General Risk Measures

<!--
INSTRUCTIONS FOR WRITING A CLEAR README:
1. Be concise but comprehensive: Explain what the project does in the first paragraph.
2. Guide the user: Provide step-by-step instructions on how to run the pipeline from raw data to final results.
3. Explain the structure: Help readers (and your advisor) navigate your codebase easily.
4. Keep it updated: As your thesis evolves, update the dependencies and run commands.
5. Document inputs/outputs: Clearly state what data is expected in `data/raw/` and what outputs will land in `plots/` or `data/processed/`.
-->

## Overview

The repository contains the codebase for my MSc Thesis in the program Quantitative Finance at ETH Zürich and University of Zürich. It implements and compares two international portfolio optimization papers under Mean Variance and Expected Shortfall by Ulyrch, Lucescu and Burkhardt (Accessible on ScienceDirect, "Sparse and stable international portfolio optimization and currency risk management" https://www.sciencedirect.com/science/article/pii/S026156062300150X, TODO second paper) and extends the optimization to general risk measures by introducing TODO. 


## Repository Structure

- `administrative/`: Documents related to the thesis proposal and registration.
- `code/`: Contains all the Python codebase for the thesis.
  - `scripts/`: Executable scripts to run specific experiments and tests (e.g., `base_script.py` for running the main optimization script, `return_statistics.py` for computing return statistics).
  - `src/`: Core source code modules.
    - `backtest/`: The core `Backtester` engine designed for the `BaseOptimizer` class.
    - `data_pipeline/`: Scripts to clean, format, and preprocess raw data.
    - `evaluation/`: Scripts for evaluating backtest performance and computing metrics.
    - `optimizers/`: Implementations of various portfolio optimizers (e.g., `JointMeanVarianceOptimizer`, `EqualWeightOptimizer`) following the `BaseOptimizer` class structure.
- `data/`: Datasets used in the project (see `data/README_DATA.md` for details). Some data may be absent from the repository due to licensing constraints.
  - `raw/`: Unaltered downloaded data from Refinitiv Datastream and Jupyter notebooks for downloading data.
  - `interim/`: Intermediate data that has been transformed by the data pipeline.
  - `processed/`: Final datasets for modeling and backtesting.
- `manuscript/`: LaTeX source files for the thesis document, including chapters, bibliography, and compiled PDFs.
- `plots/`: Output directory for generated visualizations.

## Installation & Setup

```bash
# 1. Clone the repository
git clone <repository_url>
cd master_thesis

# 2. Set up a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# 3. Install dependencies
# TODO Make sure to generate a requirements.txt: `pip freeze > requirements.txt`
pip install -r requirements.txt
```

## Usage

### 0. Datastream Download
Due to licensing constraints, the raw data used for this project cannot be distributed publicly on GitHub. To reproduce the initial datasets, you will need access to Refinitiv Datastream and a configured credentials file.


### 1. Data Processing
To generate the processed datasets used in the analysis from the raw inputs run the following scripts in order. 
```bash
python code/src/data_pipeline/format_raw_data.py
python code/src/data_pipeline/preprocess_data.py
```

### 2. Running Optimizations & Backtests
An example of how to execute the main backtesting scripts to obtain the return series of the optimized portfolios.
```bash
python code/scripts/base_script.py
```

## Results & Evaluation
Running the scripts will generate performance metrics and visualizations which are automatically saved to the `plots/` directory.

## Author
**Anej Rozman**  
MSc Quantitative Finance, 
ETH Zürich, Univerity of Zürich
