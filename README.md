# Portfolio Optimization Project

This repository contains the implementation of various portfolio optimization strategies for a quantitative finance project, including the Fast Algorithm (FA) benchmark as described in Modern Portfolio Theory.

## Project Overview

This project is an implementation of an investment strategy response to a Request for Proposal (RFP) following the guidelines of the 553.647: Quantitative Portfolio Theory and Performance Analysis course. The project aims to:

1. Create and implement a portfolio optimization strategy
2. Back-test the strategy against historical ETF data
3. Compare performance against conventional benchmarks:
   - S&P 500 (SPY)
   - 60/40 Stock/Bond portfolio
   - Markowitz MPT long-only portfolio (Fast Algorithm)

## Repository Structure

```
portfolio-optimization/
├── data/                      # Data directory
│   ├── raw/                   # Raw unprocessed data
│   └── processed/             # Processed data ready for analysis
├── notebooks/                 # Jupyter notebooks for exploration and reporting
├── results/                   # Results directory
│   ├── figures/               # Generated visualizations
│   └── models/                # Saved model parameters
├── src/                       # Source code
│   ├── analysis/              # Analysis of portfolios
│   ├── preprocessing/         # Data preprocessing scripts
│   └── optimization/          # Portfolio optimization algorithms
├── .gitignore                 # Git ignore file
├── pyproject.toml             # Python project configuration
└── README.md                  # Project documentation
```

## Setup and Installation

### Prerequisites

- Python 3.8+
- UV package manager

### Setting up the environment

```bash
# Clone the repository
git clone https://github.com/yourusername/portfolio-optimization.git
cd portfolio-optimization

# Create a virtual environment
uv venv portfolio_env

# Activate the virtual environment
# On Windows:
portfolio_env\Scripts\activate
# On macOS/Linux:
source portfolio_env/bin/activate

# Install dependencies
uv pip install -e .
```

### Optional development dependencies

```bash
# Install development dependencies
uv pip install -e ".[dev]"
```

## Usage

### Data Preprocessing

```bash
# Run the preprocessing script
python -m src.preprocessing.process_etf_data --input data/raw/project_sample.csv --output data/processed/
```

### Portfolio Optimization

```bash
# Run the Fast Algorithm benchmark
python -m src.optimization.fast_algorithm --data data/processed/sector_returns.csv --period "2019-2023" --output results/models/fa_weights.csv
```

### Visualization

```bash
# Run the visualization script
python -m src.optimization.visualize_portfolio --weights results/models/fa_weights.csv --returns data/processed/sector_returns.csv --output results/figures/
```

## Implemented Strategies

1. **Fast Algorithm (FA) Benchmark**: Implementation of the Markowitz Mean-Variance Optimization with long-only constraints
2. **60/40 Stock/Bond Strategy**: Traditional allocation with SPY and bond ETFs
3. **Custom Strategy**: [Brief description of your custom strategy]

## Performance Metrics

The portfolio evaluation includes the following metrics:

- Total Return
- Annualized Volatility
- Sharpe Ratio
- Maximum Drawdown
- Value at Risk (VaR)
- Conditional Value at Risk (CVaR)
- Rolling 1-year returns

## License

This project is licensed under the MIT License - see the LICENSE file for details.
