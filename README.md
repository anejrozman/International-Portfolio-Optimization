# International Portfolio Optimization Under General Risk Measures

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

- `data/`: Not present in the repository due to licensing constraints, but this is where all datasets are organized.
  - `raw/`: Unaltered downloaded data from Refinitiv Datastream..
  - `interim/`: Intermediate data that has been transformed by the data_pipeline
  - `processed/`: Final datasets for modeling and backtesting.
- `src/`: Core source code modules.
  - `data_pipeline/`: Scripts to clean, format, and preprocess raw data.
  - `optimizers/`: Implementations of various portfolio optimizers inherited from `BaseOptimizer` (e.g., `JointMeanVarianceOptimizer`, `EqualWeightOptimizer`).
  - `backtest/`: The core `Backtester` engine designed for the `BaseOptimizer` class.
  - `evaluation/`: Scripts for evaluating backtest performance and computing metrics.
- `scripts/`: Executable scripts to run specific experiments and tests.
- `plots/`: Output directory for generated visualizations.

## Installation & Setup
TODO 

```bash
# 1. Clone the repository
git clone TODO
cd thesis_codebase

# 2. Set up a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# 3. Install dependencies
# Make sure to generate a requirements.txt: `pip freeze > requirements.txt`
pip install -r requirements.txt
```

## Usage

### 0. Datastream Download
Due to licensing constraints, the raw data used for this project cannot be distributed publicly on GitHub. To reproduce the initial datasets, you will need access to Refinitiv Datastream and a configured credentials file.


### 1. Data Processing
To generate the processed datasets used in the analysis from the raw inputs run the following scripts in order. 
```bash
python src/data_pipeline/format_raw_data.py
python src/data_pipeline/preprocess_data.py
```

### 2. Running Optimizations & Backtests
An example of how to execute the main backtesting scripts to obtain the return series of the optimized portfolios.
```bash
python scripts/base_script.py
```

## Results & Evaluation
Running the scripts will generate performance metrics and visualizations which are automatically saved to the `plots/` directory.

## Author
**Anej Rozman**  
MSc Quantitative Finance, 
ETH Zürich, Univerity of Zürich
