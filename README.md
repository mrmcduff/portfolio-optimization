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
git clone https://github.com/mrmcduff/portfolio-optimization.git
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

# Using the Run Script

The run.py script provides a convenient command-line interface for executing all aspects of the portfolio optimization project. This tool streamlines common tasks by setting sensible defaults and handling the complex command arguments for you.

## Basic Usage

The script supports several commands that correspond to different steps in the portfolio optimization workflow:

```bash
# Make the script executable
chmod +x run.py

# View available commands
./run.py --help
```

## Available Commands

### Data Preprocessing

Preprocess the raw ETF data to calculate returns and prepare it for analysis:

```bash
./run.py preprocess
```

By default, this reads from `data/raw/course_project_data.csv` and outputs to `data/processed/.`

### Generating Benchmarks

Create standard benchmark returns for comparison:

```bash
./run.py benchmarks
```

This generates the S&P 500 (SPY) and 60/40 portfolio benchmarks with monthly rebalancing.

### Portfolio Optimization

Run the Fast Algorithm to create an optimized portfolio:

```bash
./run.py optimize --period recent
```

The `--period` parameter can be:
- `financial_crisis`: Focus on the 2007-2009 financial crisis period
- `post_crisis`: Focus on the recovery period after the financial crisis
- `recent`: Focus on the most recent market data
- `custom`: Specify your own custom date range

#### Custom Date Ranges

You can analyze any specific time period by using the custom period option:

```bash
# Analyze a specific 5-year period
./run.py optimize --period custom --start-date 2015-01-01 --years 5

# Analyze a specific date range
./run.py optimize --period custom --start-date 2018-01-01 --end-date 2020-12-31
```

When using custom periods, the system automatically:
- Filters to the specified date range
- Excludes ETFs that don't have sufficient data during that period (< 80% coverage)
- Calculates returns for the valid ETFs

### Visualizing Results

Generate visualizations of your portfolio performance:

```
./run.py visualize
```

This creates charts comparing your optimized portfolio against benchmarks and showing asset allocations.

### Analyzing Performance

Perform comprehensive performance analysis:

```bash
./run.py analyze
```

This generates detailed performance metrics and creates a summary report.

### Comparing Rebalancing Strategies

Compare different portfolio rebalancing approaches:

```bash
./run.py rebalancing
```

This analyzes how various rebalancing methods (monthly, quarterly, threshold-based) affect portfolio performance.

### Running the Complete Pipeline

Execute all steps in sequence with a single command:

```bash
./run.py all
```

This runs the entire workflow from raw data to final analysis, using sensible defaults throughout.

### Customizing Commands

Each command accepts parameters to customize its behavior:

```bash
# Specify custom input and output locations
./run.py preprocess --input custom_data.csv --output custom_output_dir/

# Set a different risk-free rate for optimization
./run.py optimize --risk-free-rate 0.03 --period recent

# Customize the rebalancing analysis
./run.py rebalancing --threshold 0.1 --stock SPY --bond BND
```

For a complete list of options for each command, use:

```bash
./run.py [command] --help
```

### Default File Paths

The script uses these default paths unless overridden:

- Raw data: `data/raw/course_project_data.csv`
- Processed data: `data/processed/`
- Model outputs: `results/models/`
- Visualizations: `results/figures/`
- Analysis reports: `results/analysis/`

### Project Directory Structure

When you run the script, it automatically creates any necessary directories following this structure:

```
portfolio-optimization/
├── data/                      # Data directory
│   ├── raw/                   # Raw unprocessed data
│   └── processed/             # Processed data ready for analysis
├── results/                   # Results directory
│   ├── figures/               # Generated visualizations
│   ├── models/                # Saved model parameters
│   └── analysis/              # Performance analysis reports
└── run.py                     # The convenience script
```

This command-line interface significantly simplifies the workflow for portfolio optimization, allowing you to focus on analyzing results rather than managing complex command arguments.

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
