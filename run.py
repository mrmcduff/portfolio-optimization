#!/usr/bin/env python
"""
Portfolio Optimization Project Runner

This script provides convenience commands to run the various components
of the portfolio optimization project with sensible defaults.
"""

import argparse
import os
import subprocess
import sys
from typing import List

from src.analysis import benchmark_analysis

# Default file paths
DEFAULT_PATHS = {
    # Input data
    "raw_data": "data/raw/course_project_data.csv",
    # Processed data
    "processed_dir": "data/processed",
    "daily_returns": "data/processed/daily_returns.csv",
    "sector_returns": "data/processed/sector_returns.csv",
    "spy_returns": "data/processed/spy_returns.csv",
    "balanced_returns": "data/processed/balanced_returns.csv",
    # Results
    "models_dir": "results/models",
    "ca_weights": "results/models/fa_optimal_weights.csv",
    "ca_returns": "results/models/fa_portfolio_returns.csv",
    "ra_returns": "results/models/returns_algorithm_portfolio.csv",  # Returns algorithm portfolio returns
    "wtf_returns": "results/models/weighted_top_five_portfolio.csv",  # Weighted Top Five portfolio returns
    # Visualizations
    "figures_dir": "results/figures",
    # Analysis
    "analysis_dir": "results/analysis",
}


def ensure_directory_exists(directory_path: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.

    Parameters:
    -----------
    directory_path : str
        Path to the directory to check/create
    """
    if not os.path.exists(directory_path):
        print(f"Creating directory: {directory_path}")
        os.makedirs(directory_path, exist_ok=True)


def run_command(command: List[str], display_command: bool = True) -> int:
    """
    Run a shell command and display its output.

    Parameters:
    -----------
    command : List[str]
        Command and arguments to run
    display_command : bool, optional
        Whether to display the command being run, by default True

    Returns:
    --------
    int
        Exit code from the command
    """
    if display_command:
        print(f"\n=== Running: {' '.join(command)} ===\n")

    try:
        # Run the command and stream output in real-time
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Stream output in real-time
        for line in iter(process.stdout.readline, ""):
            print(line, end="")

        # Wait for process to complete and get exit code
        return_code = process.wait()

        if return_code != 0:
            print(f"\n=== Command failed with exit code {return_code} ===\n")
        else:
            print("\n=== Command completed successfully ===\n")

        return return_code

    except Exception as e:
        print(f"Error running command: {e}")
        return 1


def preprocess(args: argparse.Namespace) -> int:
    """
    Run data preprocessing.

    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments

    Returns:
    --------
    int
        Exit code from the command
    """
    # Override defaults with any provided args
    input_file = getattr(args, "input", None) or DEFAULT_PATHS["raw_data"]
    output_dir = getattr(args, "output", None) or DEFAULT_PATHS["processed_dir"]

    # Ensure output directory exists
    ensure_directory_exists(output_dir)

    # Build and run the command
    command = [
        "python",
        "-m",
        "src.preprocessing.process_etf_data",
        "--input",
        input_file,
        "--output",
        output_dir,
    ]

    return run_command(command)


def generate_benchmarks(args: argparse.Namespace) -> int:
    """
    Generate benchmark returns.

    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments

    Returns:
    --------
    int
        Exit code from the command
    """
    # Override defaults with any provided args
    returns_file = getattr(args, "returns", None) or DEFAULT_PATHS["daily_returns"]
    output_dir = getattr(args, "output", None) or DEFAULT_PATHS["processed_dir"]

    # Ensure output directory exists
    ensure_directory_exists(output_dir)

    # Build and run the command
    command = [
        "python",
        "-m",
        "src.preprocessing.generate_benchmarks",
        "--returns",
        returns_file,
        "--output",
        output_dir,
    ]

    # Add optional parameters if provided
    if hasattr(args, "stock") and args.stock:
        command.extend(["--stock", args.stock])
    if hasattr(args, "bond") and args.bond:
        command.extend(["--bond", args.bond])
    if hasattr(args, "weight") and args.weight:
        command.extend(["--weight", str(args.weight)])

    # Add benchmark rebalancing parameters
    if hasattr(args, "rebalance") and args.rebalance:
        command.extend(["--rebalance", args.rebalance])
    if hasattr(args, "frequency") and args.frequency:
        command.extend(["--frequency", args.frequency])
    if hasattr(args, "threshold") and args.threshold:
        command.extend(["--threshold", str(args.threshold)])

    return run_command(command)


def optimize(args: argparse.Namespace) -> int:
    """
    Run the Custom Algorithm portfolio optimization.

    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments

    Returns:
    --------
    int
        Exit code from the command
    """
    # Override defaults with any provided args
    data_file = getattr(args, "data", None) or DEFAULT_PATHS["sector_returns"]
    output_dir = getattr(args, "output", None) or DEFAULT_PATHS["models_dir"]

    # Ensure output directory exists
    ensure_directory_exists(output_dir)

    # Build and run the command
    command = [
        "python",
        "-m",
        "src.optimization.custom_algorithm",
        "--data",
        data_file,
        "--output",
        output_dir,
    ]

    # Add optional parameters if provided
    if hasattr(args, "period") and args.period:
        command.extend(["--period", args.period])

        # Add custom period parameters if using custom period
        if args.period == "custom":
            if hasattr(args, "start_date") and args.start_date:
                command.extend(["--start-date", args.start_date])
            if hasattr(args, "end_date") and args.end_date:
                command.extend(["--end-date", args.end_date])
            if hasattr(args, "years") and args.years:
                command.extend(["--years", str(args.years)])

    if hasattr(args, "risk_free_rate") and args.risk_free_rate:
        command.extend(["--risk-free-rate", str(args.risk_free_rate)])
    if hasattr(args, "max_weight") and args.max_weight:
        command.extend(["--max-weight", str(args.max_weight)])
    if hasattr(args, "min_weight") and args.min_weight:
        command.extend(["--min-weight", str(args.min_weight)])
    if hasattr(args, "allow_short") and args.allow_short:
        command.append("--allow-short")

    # Add Custom Algorithm rebalancing parameters
    if hasattr(args, "rebalance_portfolio") and args.rebalance_portfolio:
        command.append("--rebalance-portfolio")
    if hasattr(args, "rebalance_frequency") and args.rebalance_frequency:
        command.extend(["--rebalance-frequency", args.rebalance_frequency])
    if hasattr(args, "estimation_window") and args.estimation_window:
        command.extend(["--estimation-window", str(args.estimation_window)])

    return run_command(command)


def visualize(args: argparse.Namespace) -> int:
    """
    Run portfolio visualization.

    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments

    Returns:
    --------
    int
        Exit code from the command
    """
    # Override defaults with any provided args
    weights_file = getattr(args, "weights", None) or DEFAULT_PATHS["ca_weights"]
    returns_file = getattr(args, "returns", None) or DEFAULT_PATHS["sector_returns"]
    output_dir = getattr(args, "output", None) or DEFAULT_PATHS["figures_dir"]

    # Ensure output directory exists
    ensure_directory_exists(output_dir)

    # Build and run the command
    command = [
        "python",
        "-m",
        "src.optimization.visualize_portfolio",
        "--weights",
        weights_file,
        "--returns",
        returns_file,
        "--output",
        output_dir,
    ]

    # Add optional parameters if provided
    if hasattr(args, "spy_benchmark") and args.spy_benchmark:
        command.extend(["--spy-benchmark", args.spy_benchmark])
    elif os.path.exists(DEFAULT_PATHS["spy_returns"]):
        command.extend(["--spy-benchmark", DEFAULT_PATHS["spy_returns"]])

    if hasattr(args, "bond_benchmark") and args.bond_benchmark:
        command.extend(["--bond-benchmark", args.bond_benchmark])
    elif os.path.exists(DEFAULT_PATHS["balanced_returns"]):
        command.extend(["--bond-benchmark", DEFAULT_PATHS["balanced_returns"]])

    if hasattr(args, "risk_free_rate") and args.risk_free_rate:
        command.extend(["--risk-free-rate", str(args.risk_free_rate)])

    return run_command(command)


def analyze(args: argparse.Namespace) -> int:
    """
    Run benchmark analysis on portfolio returns.

    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments

    Returns:
    --------
    int
        Exit code from the command
    """
    # Get default files if none specified
    if not args.files:
        default_files = [
            DEFAULT_PATHS["spy_returns"],
            DEFAULT_PATHS["balanced_returns"],
            DEFAULT_PATHS["ca_returns"],
        ]
        if os.path.exists(DEFAULT_PATHS["ra_returns"]):
            default_files.append(DEFAULT_PATHS["ra_returns"])
        if os.path.exists(DEFAULT_PATHS["wtf_returns"]):
            default_files.append(DEFAULT_PATHS["wtf_returns"])
        args.files = default_files

    # Get default names if none specified
    if not args.names:
        if len(args.files) == 5:
            args.names = [
                "S&P 500",
                "60/40 Portfolio",
                "Custom Algorithm Portfolio",
                "Returns Algorithm Portfolio",
                "Weighted Top Five Portfolio",
            ]
        elif len(args.files) == 4:
            args.names = [
                "S&P 500",
                "60/40 Portfolio",
                "Custom Algorithm Portfolio",
                "Returns Algorithm Portfolio",
            ]
        elif len(args.files) == 3:
            args.names = [
                "S&P 500",
                "60/40 Portfolio",
                "Custom Algorithm Portfolio",
            ]
        else:
            args.names = [f"Portfolio {i + 1}" for i in range(len(args.files))]

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Run benchmark analysis
    try:
        benchmark_analysis.main(
            return_files=args.files,
            names=args.names,
            risk_free_rate=args.risk_free_rate,
            output_dir=args.output,
        )
        return 0
    except Exception as e:
        print(f"Error running benchmark analysis: {e}")
        return 1


def compare_rebalancing(args: argparse.Namespace) -> int:
    """
    Compare different portfolio rebalancing strategies.

    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments

    Returns:
    --------
    int
        Exit code from the command
    """
    # Override defaults with any provided args
    returns_file = getattr(args, "returns", None) or DEFAULT_PATHS["daily_returns"]
    output_dir = getattr(args, "output", None) or os.path.join(
        DEFAULT_PATHS["analysis_dir"], "rebalancing"
    )

    # Ensure output directory exists
    ensure_directory_exists(output_dir)

    # Build and run the command
    command = [
        "python",
        "-m",
        "src.analysis.compare_rebalancing",
        "--returns",
        returns_file,
        "--output",
        output_dir,
    ]

    # Add optional parameters
    if hasattr(args, "stock") and args.stock:
        command.extend(["--stock", args.stock])
    if hasattr(args, "bond") and args.bond:
        command.extend(["--bond", args.bond])
    if hasattr(args, "weight") and args.weight:
        command.extend(["--weight", str(args.weight)])
    if hasattr(args, "risk_free_rate") and args.risk_free_rate:
        command.extend(["--risk-free-rate", str(args.risk_free_rate)])
    if hasattr(args, "threshold") and args.threshold:
        command.extend(["--threshold", str(args.threshold)])

    return run_command(command)


def run_returns_algorithm(args: argparse.Namespace) -> int:
    """
    Run the Returns Algorithm portfolio optimization.

    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments

    Returns:
    --------
    int
        Exit code from the command
    """
    # Override defaults with any provided args
    data_file = getattr(args, "data", None) or DEFAULT_PATHS["sector_returns"]
    output_dir = getattr(args, "output", None) or DEFAULT_PATHS["models_dir"]

    # Ensure output directory exists
    ensure_directory_exists(output_dir)

    # Build and run the command
    command = [
        "python",
        "-m",
        "src.optimization.returns_algorithm",
        "--returns",
        data_file,
        "--output",
        output_dir,
    ]

    # Add optional parameters if provided
    if hasattr(args, "rebalance_frequency") and args.rebalance_frequency:
        command.extend(["--rebalance", args.rebalance_frequency])
    if hasattr(args, "lookback_window") and args.lookback_window:
        command.extend(["--lookback", str(args.lookback_window)])
    if hasattr(args, "max_weight") and args.max_weight:
        command.extend(["--max-weight", str(args.max_weight)])

    return run_command(command)


def run_weighted_top_five(args: argparse.Namespace) -> int:
    """
    Run the Weighted Top Five portfolio optimization.

    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments

    Returns:
    --------
    int
        Exit code from the command
    """
    # Override defaults with any provided args
    data_file = getattr(args, "data", None) or DEFAULT_PATHS["sector_returns"]
    output_dir = getattr(args, "output", None) or DEFAULT_PATHS["models_dir"]

    # Ensure output directory exists
    ensure_directory_exists(output_dir)

    # Build and run the command
    command = [
        "python",
        "-m",
        "src.optimization.weighted_top_five",
        "--returns",
        data_file,
        "--output",
        output_dir,
    ]

    # Add optional parameters if provided
    if hasattr(args, "rebalance_frequency") and args.rebalance_frequency:
        command.extend(["--rebalance", args.rebalance_frequency])
    if hasattr(args, "lookback_window") and args.lookback_window:
        command.extend(["--lookback", str(args.lookback_window)])

    return run_command(command)


def run_all(args: argparse.Namespace) -> int:
    """
    Run all steps of the portfolio optimization pipeline.

    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments

    Returns:
    --------
    int
        Exit code from the command
    """
    # Create default args for each step
    preprocess_args = argparse.Namespace(
        input=getattr(args, "input", None), output=DEFAULT_PATHS["processed_dir"]
    )

    benchmark_args = argparse.Namespace(
        returns=DEFAULT_PATHS["daily_returns"],
        output=DEFAULT_PATHS["processed_dir"],
        stock=None,
        bond=None,
        weight=None,
        rebalance="periodic",
        frequency="M",
        threshold=0.05,
    )

    optimize_args = argparse.Namespace(
        data=DEFAULT_PATHS["sector_returns"],
        output=DEFAULT_PATHS["models_dir"],
        period=getattr(args, "period", "recent"),
        risk_free_rate=getattr(args, "risk_free_rate", 0.02),
        max_weight=0.25,
        min_weight=0.01,
        allow_short=False,
        rebalance_portfolio=getattr(args, "rebalance_portfolio", False),
        rebalance_frequency=getattr(args, "rebalance_frequency", "Q"),
        estimation_window=getattr(args, "estimation_window", 252),
    )

    returns_args = argparse.Namespace(
        data=DEFAULT_PATHS["sector_returns"],
        output=DEFAULT_PATHS["models_dir"],
        rebalance_frequency="Q",
        lookback_window=63,
        max_weight=0.2,
    )

    weighted_args = argparse.Namespace(
        data=DEFAULT_PATHS["sector_returns"],
        output=DEFAULT_PATHS["models_dir"],
        rebalance_frequency="Q",
        lookback_window=63,
    )

    visualize_args = argparse.Namespace(
        weights=DEFAULT_PATHS["ca_weights"],
        returns=DEFAULT_PATHS["sector_returns"],
        output=DEFAULT_PATHS["figures_dir"],
        spy_benchmark=DEFAULT_PATHS["spy_returns"],
        bond_benchmark=DEFAULT_PATHS["balanced_returns"],
        risk_free_rate=getattr(args, "risk_free_rate", 0.02),
    )

    analyze_args = argparse.Namespace(
        files=[
            DEFAULT_PATHS["spy_returns"],
            DEFAULT_PATHS["balanced_returns"],
            DEFAULT_PATHS["ca_returns"],
        ],
        names=[
            "S&P 500",
            "60/40 Portfolio",
            "Custom Algorithm Portfolio",
        ],
        risk_free_rate=getattr(args, "risk_free_rate", 0.02),
        output=DEFAULT_PATHS["analysis_dir"],
    )

    # Run each step in sequence with its specific arguments
    steps = [
        ("Preprocessing", preprocess, preprocess_args),
        ("Benchmark Generation", generate_benchmarks, benchmark_args),
        ("Custom Algorithm Optimization", optimize, optimize_args),
        ("Visualization", visualize, visualize_args),
        ("Benchmark Analysis", analyze, analyze_args),
    ]

    # Add optional algorithms if explicitly requested
    if getattr(args, "include_returns", False):
        steps.insert(
            3, ("Returns Algorithm Optimization", run_returns_algorithm, returns_args)
        )
        analyze_args.files.append(DEFAULT_PATHS["ra_returns"])
        analyze_args.names.append("Returns Algorithm Portfolio")

    if getattr(args, "include_weighted", False):
        steps.insert(
            4, ("Weighted Top Five Optimization", run_weighted_top_five, weighted_args)
        )
        analyze_args.files.append(DEFAULT_PATHS["wtf_returns"])
        analyze_args.names.append("Weighted Top Five Portfolio")

    for step_name, step_func, step_args in steps:
        print(f"\n{'=' * 80}\n=== Running Step: {step_name} ===\n{'=' * 80}\n")
        result = step_func(step_args)
        if result != 0:
            print(f"\nError in {step_name} step. Stopping pipeline.")
            return result
        print(f"\n{step_name} step completed successfully.\n")

    print("\nAll steps completed successfully!")
    return 0


def main() -> int:
    """
    Main entry point for the script.

    Returns:
    --------
    int
        Exit code
    """
    # Create main parser
    parser = argparse.ArgumentParser(
        description="Portfolio Optimization Project Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Preprocess command
    preprocess_parser = subparsers.add_parser(
        "preprocess", help="Run data preprocessing"
    )
    preprocess_parser.add_argument(
        "--input", "-i", help=f"Input data file (default: {DEFAULT_PATHS['raw_data']})"
    )
    preprocess_parser.add_argument(
        "--output",
        "-o",
        help=f"Output directory (default: {DEFAULT_PATHS['processed_dir']})",
    )

    # Generate benchmarks command
    benchmark_parser = subparsers.add_parser(
        "benchmarks", help="Generate benchmark returns"
    )
    benchmark_parser.add_argument(
        "--returns",
        "-r",
        help=f"Returns data file (default: {DEFAULT_PATHS['daily_returns']})",
    )
    benchmark_parser.add_argument(
        "--output",
        "-o",
        help=f"Output directory (default: {DEFAULT_PATHS['processed_dir']})",
    )
    benchmark_parser.add_argument(
        "--stock", "-s", help="Stock ETF symbol (default: SPY)"
    )
    benchmark_parser.add_argument("--bond", "-b", help="Bond ETF symbol (default: BND)")
    benchmark_parser.add_argument(
        "--weight",
        "-w",
        type=float,
        help="Weight for stocks in balanced portfolio (default: 0.6)",
    )
    # Add rebalancing options for benchmarks
    benchmark_parser.add_argument(
        "--rebalance",
        choices=["none", "periodic", "threshold"],
        default="periodic",
        help="Rebalancing method for benchmarks (default: periodic)",
    )
    benchmark_parser.add_argument(
        "--frequency",
        choices=["D", "W", "M", "Q", "A"],
        default="M",
        help="Rebalancing frequency for periodic method (default: M for monthly)",
    )
    benchmark_parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Rebalancing threshold for threshold method (default: 0.05 or 5%%)",
    )

    # Optimize command
    optimize_parser = subparsers.add_parser(
        "optimize", help="Run Custom Algorithm portfolio optimization"
    )
    optimize_parser.add_argument(
        "--data",
        "-d",
        help=f"Returns data file (default: {DEFAULT_PATHS['sector_returns']})",
    )
    optimize_parser.add_argument(
        "--output",
        "-o",
        help=f"Output directory (default: {DEFAULT_PATHS['models_dir']})",
    )
    optimize_parser.add_argument(
        "--period",
        "-p",
        choices=["financial_crisis", "post_crisis", "recent", "custom"],
        help="Period to analyze (use 'custom' with --start-date and --end-date for custom periods)",
    )
    optimize_parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for custom period (YYYY-MM-DD format)",
    )
    optimize_parser.add_argument(
        "--end-date",
        type=str,
        help="End date for custom period (YYYY-MM-DD format)",
    )
    optimize_parser.add_argument(
        "--years",
        type=int,
        help="Number of years for custom period (starting from start-date)",
    )
    optimize_parser.add_argument(
        "--risk-free-rate", "-r", type=float, help="Risk-free rate (annualized)"
    )
    optimize_parser.add_argument(
        "--max-weight", "-m", type=float, help="Maximum weight for any asset"
    )
    optimize_parser.add_argument(
        "--min-weight", type=float, help="Minimum weight for any asset if included"
    )
    optimize_parser.add_argument(
        "--allow-short", action="store_true", help="Allow short selling (not long-only)"
    )
    # Add rebalancing options for Custom Algorithm
    optimize_parser.add_argument(
        "--rebalance-portfolio",
        action="store_true",
        help="Enable periodic rebalancing for the Custom Algorithm portfolio",
    )
    optimize_parser.add_argument(
        "--rebalance-frequency",
        choices=["D", "W", "M", "Q", "A"],
        default="Q",
        help="Frequency for portfolio rebalancing (default: Q for quarterly)",
    )
    optimize_parser.add_argument(
        "--estimation-window",
        type=int,
        default=252,
        help="Lookback window in trading days for parameter estimation (default: 252 days)",
    )

    # Visualize command
    visualize_parser = subparsers.add_parser(
        "visualize", help="Run portfolio visualization"
    )
    visualize_parser.add_argument(
        "--weights", "-w", help=f"Weights file (default: {DEFAULT_PATHS['ca_weights']})"
    )
    visualize_parser.add_argument(
        "--returns",
        "-r",
        help=f"Returns data file (default: {DEFAULT_PATHS['sector_returns']})",
    )
    visualize_parser.add_argument(
        "--output",
        "-o",
        help=f"Output directory (default: {DEFAULT_PATHS['figures_dir']})",
    )
    visualize_parser.add_argument(
        "--spy-benchmark", help="Path to S&P 500 (SPY) returns CSV"
    )
    visualize_parser.add_argument(
        "--bond-benchmark", help="Path to 60/40 portfolio returns CSV"
    )
    visualize_parser.add_argument(
        "--risk-free-rate", type=float, help="Risk-free rate (annualized)"
    )

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Run benchmark analysis")
    analyze_parser.add_argument(
        "--files", "-f", nargs="+", help="Paths to CSV files containing return data"
    )
    analyze_parser.add_argument(
        "--names", "-n", nargs="+", help="Names for each return series"
    )
    analyze_parser.add_argument(
        "--risk-free-rate", "-r", type=float, help="Annualized risk-free rate"
    )
    analyze_parser.add_argument(
        "--output",
        "-o",
        help=f"Output directory (default: {DEFAULT_PATHS['analysis_dir']})",
    )

    # Rebalancing comparison command
    rebalance_parser = subparsers.add_parser(
        "rebalancing", help="Compare different portfolio rebalancing strategies"
    )
    rebalance_parser.add_argument(
        "--returns",
        "-r",
        help=f"Returns data file (default: {DEFAULT_PATHS['daily_returns']})",
    )
    rebalance_parser.add_argument("--output", "-o", help="Directory to save results")
    rebalance_parser.add_argument(
        "--stock", "-s", default="SPY", help="Stock ETF symbol"
    )
    rebalance_parser.add_argument("--bond", "-b", default="BND", help="Bond ETF symbol")
    rebalance_parser.add_argument(
        "--weight", "-w", type=float, default=0.6, help="Target stock weight"
    )
    rebalance_parser.add_argument(
        "--risk-free-rate", type=float, default=0.02, help="Risk-free rate (annualized)"
    )
    rebalance_parser.add_argument(
        "--threshold", "-t", type=float, default=0.05, help="Rebalancing threshold"
    )

    # Returns Algorithm command
    returns_parser = subparsers.add_parser(
        "returns", help="Run Returns Algorithm portfolio optimization"
    )
    returns_parser.add_argument(
        "--data",
        "-d",
        help=f"Returns data file (default: {DEFAULT_PATHS['sector_returns']})",
    )
    returns_parser.add_argument(
        "--output",
        "-o",
        help=f"Output directory (default: {DEFAULT_PATHS['models_dir']})",
    )
    returns_parser.add_argument(
        "--rebalance-frequency",
        choices=["D", "W", "M", "Q", "A"],
        default="Q",
        help="Rebalancing frequency (default: Q for quarterly)",
    )
    returns_parser.add_argument(
        "--lookback-window",
        type=int,
        default=63,
        help="Lookback window in days for returns calculation (default: 63 days)",
    )
    returns_parser.add_argument(
        "--max-weight",
        type=float,
        default=0.2,
        help="Maximum weight for any asset (default: 0.2 or 20%)",
    )

    # Run all command
    all_parser = subparsers.add_parser("all", help="Run all steps in sequence")
    # We'll use the same arguments for each step when running all
    all_parser.add_argument(
        "--input", "-i", help=f"Input data file (default: {DEFAULT_PATHS['raw_data']})"
    )
    all_parser.add_argument(
        "--period",
        "-p",
        choices=["financial_crisis", "post_crisis", "recent"],
        default="recent",
        help="Period to analyze (default: recent)",
    )
    all_parser.add_argument(
        "--risk-free-rate",
        "-r",
        type=float,
        default=0.02,
        help="Risk-free rate (annualized) (default: 0.02)",
    )
    # Add rebalancing options for the all command
    all_parser.add_argument(
        "--rebalance-portfolio",
        action="store_true",
        help="Enable periodic rebalancing for the Custom Algorithm portfolio",
    )
    all_parser.add_argument(
        "--rebalance-frequency",
        choices=["D", "W", "M", "Q", "A"],
        default="Q",
        help="Frequency for portfolio rebalancing (default: Q for quarterly)",
    )
    all_parser.add_argument(
        "--estimation-window",
        type=int,
        default=252,
        help="Lookback window in trading days for parameter estimation (default: 252 days)",
    )
    # Add flags for optional algorithms
    all_parser.add_argument(
        "--include-returns",
        action="store_true",
        help="Include the Returns Algorithm in the analysis",
    )
    all_parser.add_argument(
        "--include-weighted",
        action="store_true",
        help="Include the Weighted Top Five Algorithm in the analysis",
    )

    # Parse arguments
    args = parser.parse_args()

    # If no command specified, run 'all' by default
    if args.command is None:
        return run_all(args)

    # Call the appropriate function
    command_funcs = {
        "preprocess": preprocess,
        "benchmarks": generate_benchmarks,
        "optimize": optimize,
        "returns": run_returns_algorithm,
        "visualize": visualize,
        "analyze": analyze,
        "rebalancing": compare_rebalancing,
        "all": run_all,
    }

    return command_funcs[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
