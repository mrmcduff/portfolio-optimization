"""
Benchmark Analysis Module

This script analyzes benchmark returns and generates a summary report
of key performance metrics for comparison with optimized portfolios.
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_return_files(
    file_paths: List[str], names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load multiple return files and combine them into a single DataFrame.

    Parameters:
    -----------
    file_paths : List[str]
        List of paths to CSV files containing return data
    names : Optional[List[str]], optional
        Names to use for each return series, by default None (uses column names from files)

    Returns:
    --------
    pd.DataFrame
        Combined return data
    """
    all_returns = []
    all_names = []

    for i, file_path in enumerate(file_paths):
        try:
            # Load returns from CSV
            data = pd.read_csv(file_path, index_col=0, parse_dates=True)

            # Get the returns series (first column if multiple columns)
            returns = data.iloc[:, 0] if data.shape[1] >= 1 else data

            # Determine name
            if names and i < len(names) and names[i]:
                name = names[i]
            else:
                # Use column name from file or file basename
                name = (
                    returns.name
                    if returns.name
                    else os.path.basename(file_path).split(".")[0]
                )

            # Add to lists
            all_returns.append(returns)
            all_names.append(name)

            print(f"Loaded returns from {file_path} as '{name}'")
        except Exception as e:
            print(f"Error loading returns from {file_path}: {e}")

    # Combine all return series into a DataFrame
    if all_returns:
        combined = pd.concat(all_returns, axis=1)
        combined.columns = all_names
        return combined
    else:
        raise ValueError("No valid return files were loaded")


def calculate_performance_metrics(
    returns: pd.DataFrame, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> pd.DataFrame:
    """
    Calculate key performance metrics for each return series.

    Parameters:
    -----------
    returns : pd.DataFrame
        Return data with each column representing a different portfolio/benchmark
    risk_free_rate : float, optional
        Annualized risk-free rate, by default 0.0
    periods_per_year : int, optional
        Number of periods per year (252 for daily, 12 for monthly), by default 252

    Returns:
    --------
    pd.DataFrame
        Performance metrics for each return series
    """
    # Initialize metrics dictionary
    metrics: Dict[str, Dict[str, float]] = {}

    # Daily risk-free rate
    # daily_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

    for column in returns.columns:
        series = returns[column].dropna()

        # Calculate metrics
        total_return = (1 + series).prod() - 1
        ann_return = (1 + series.mean()) ** periods_per_year - 1
        ann_volatility = series.std() * np.sqrt(periods_per_year)
        sharpe_ratio = (
            (ann_return - risk_free_rate) / ann_volatility if ann_volatility > 0 else 0
        )

        # Maximum drawdown
        cumulative_returns = (1 + series).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        max_drawdown = drawdown.min()

        # Value at Risk (95%)
        var_95 = series.quantile(0.05)

        # Skewness and kurtosis
        skewness = series.skew()
        kurtosis = series.kurtosis()

        # Store metrics
        metrics[column] = {
            "Total Return": total_return,
            "Annualized Return": ann_return,
            "Annualized Volatility": ann_volatility,
            "Sharpe Ratio": sharpe_ratio,
            "Max Drawdown": max_drawdown,
            "Value at Risk (95%)": var_95,
            "Skewness": skewness,
            "Kurtosis": kurtosis,
        }

    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics)

    return metrics_df


def plot_cumulative_returns(
    returns: pd.DataFrame, output_file: Optional[str] = None
) -> plt.Figure:
    """
    Plot cumulative returns for each return series.

    Parameters:
    -----------
    returns : pd.DataFrame
        Return data with each column representing a different portfolio/benchmark
    output_file : Optional[str], optional
        Path to save the plot, by default None

    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    # Calculate cumulative returns
    cumulative_returns = (1 + returns).cumprod()

    # Create the plot
    plt.figure(figsize=(12, 8))

    for column in cumulative_returns.columns:
        plt.plot(cumulative_returns.index, cumulative_returns[column], label=column)

    # Add labels and title
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return (1 = Initial Investment)")
    plt.title("Cumulative Return Comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add annotations for final values
    for column in cumulative_returns.columns:
        final_value = cumulative_returns[column].iloc[-1]
        plt.annotate(
            f"{final_value:.2f}x",
            xy=(cumulative_returns.index[-1], final_value),
            xytext=(5, 0),
            textcoords="offset points",
        )

    plt.tight_layout()

    # Save if output_file is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Saved cumulative returns plot to {output_file}")

    return plt.gcf()


def plot_rolling_metrics(
    returns: pd.DataFrame, window: int = 252, output_dir: Optional[str] = None
) -> List[plt.Figure]:
    """
    Plot rolling performance metrics (returns, volatility, Sharpe ratio).

    Parameters:
    -----------
    returns : pd.DataFrame
        Return data with each column representing a different portfolio/benchmark
    window : int, optional
        Rolling window size in periods, by default 252 (1 year for daily data)
    output_dir : Optional[str], optional
        Directory to save the plots, by default None

    Returns:
    --------
    List[plt.Figure]
        List of Matplotlib figures
    """
    figures = []

    # Calculate rolling metrics
    rolling_returns = returns.rolling(window=window).mean() * window
    rolling_vol = returns.rolling(window=window).std() * np.sqrt(window)
    rolling_sharpe = rolling_returns / rolling_vol

    # Plot rolling annualized returns
    fig1 = plt.figure(figsize=(12, 8))
    for column in rolling_returns.columns:
        plt.plot(rolling_returns.index, rolling_returns[column], label=column)

    plt.xlabel("Date")
    plt.ylabel(f"Rolling {window}-day Annualized Return")
    plt.title(f"Rolling {window / 252:.1f}-Year Annualized Returns")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    figures.append(fig1)

    if output_dir:
        output_file = os.path.join(output_dir, "rolling_returns.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Saved rolling returns plot to {output_file}")

    # Plot rolling volatility
    fig2 = plt.figure(figsize=(12, 8))
    for column in rolling_vol.columns:
        plt.plot(rolling_vol.index, rolling_vol[column], label=column)

    plt.xlabel("Date")
    plt.ylabel(f"Rolling {window}-day Annualized Volatility")
    plt.title(f"Rolling {window / 252:.1f}-Year Annualized Volatility")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    figures.append(fig2)

    if output_dir:
        output_file = os.path.join(output_dir, "rolling_volatility.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Saved rolling volatility plot to {output_file}")

    # Plot rolling Sharpe ratio
    fig3 = plt.figure(figsize=(12, 8))
    for column in rolling_sharpe.columns:
        plt.plot(rolling_sharpe.index, rolling_sharpe[column], label=column)

    plt.xlabel("Date")
    plt.ylabel(f"Rolling {window}-day Sharpe Ratio")
    plt.title(f"Rolling {window / 252:.1f}-Year Sharpe Ratio")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    figures.append(fig3)

    if output_dir:
        output_file = os.path.join(output_dir, "rolling_sharpe.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Saved rolling Sharpe ratio plot to {output_file}")

    return figures


def analyze_drawdowns(
    returns: pd.DataFrame, output_file: Optional[str] = None
) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Analyze drawdowns for each return series and plot underwater chart.

    Parameters:
    -----------
    returns : pd.DataFrame
        Return data with each column representing a different portfolio/benchmark
    output_file : Optional[str], optional
        Path to save the plot, by default None

    Returns:
    --------
    Tuple[pd.DataFrame, plt.Figure]
        DataFrame with drawdown statistics and Matplotlib figure
    """
    # Initialize drawdown statistics
    drawdown_stats = {}

    # Create underwater plot
    plt.figure(figsize=(12, 8))

    for column in returns.columns:
        series = returns[column].dropna()

        # Skip if series is empty
        if len(series) == 0:
            print(
                f"Warning: No data available for {column}. Skipping drawdown analysis."
            )
            drawdown_stats[column] = {
                "Max Drawdown": float("nan"),
                "Max Drawdown Date": None,
                "Recovery Time (Days)": float("nan"),
            }
            continue

        # Calculate drawdowns
        cumulative_returns = (1 + series).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1

        # Plot underwater chart
        plt.plot(drawdown.index, drawdown, label=column)

        # Calculate drawdown statistics
        max_drawdown = drawdown.min()
        # Check if drawdown is empty before calling idxmin
        if len(drawdown) > 0:
            max_drawdown_date = drawdown.idxmin()
        else:
            max_drawdown_date = None

        # Find recovery date (if any)
        if max_drawdown_date is not None and max_drawdown < 0:
            recovery_mask = (drawdown.index > max_drawdown_date) & (drawdown == 0)
            recovery_dates = drawdown.index[recovery_mask]
            recovery_date = recovery_dates[0] if len(recovery_dates) > 0 else None

            if recovery_date:
                recovery_time = (recovery_date - max_drawdown_date).days
            else:
                recovery_time = float("nan")  # Still in drawdown
        else:
            recovery_time = (
                float("nan") if max_drawdown_date is None else 0
            )  # No drawdown or no data

        # Store statistics
        drawdown_stats[column] = {
            "Max Drawdown": max_drawdown,
            "Max Drawdown Date": max_drawdown_date,
            "Recovery Time (Days)": recovery_time,
        }

    # Format plot
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.title("Portfolio Drawdowns")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save if output_file is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Saved drawdown plot to {output_file}")

    # Convert statistics to DataFrame
    drawdown_df = pd.DataFrame(drawdown_stats)

    return drawdown_df, plt.gcf()


def create_summary_report(
    returns: pd.DataFrame, metrics: pd.DataFrame, output_file: Optional[str] = None
) -> str:
    """
    Create a summary report in Markdown format.

    Parameters:
    -----------
    returns : pd.DataFrame
        Return data with each column representing a different portfolio/benchmark
    metrics : pd.DataFrame
        Performance metrics for each return series
    output_file : Optional[str], optional
        Path to save the report, by default None

    Returns:
    --------
    str
        Markdown report content
    """
    # Calculate date range
    start_date = returns.index.min().strftime("%Y-%m-%d")
    end_date = returns.index.max().strftime("%Y-%m-%d")
    total_days = len(returns)
    years = total_days / 252

    # Create report header
    report = "# Benchmark Analysis Report\n\n"
    report += f"**Analysis Period:** {start_date} to {end_date} ({total_days} trading days, ~{years:.1f} years)\n\n"

    # Add performance metrics table
    report += "## Performance Metrics\n\n"

    # Format metrics as percentage where appropriate
    formatted_metrics = metrics.copy()
    percentage_rows = [
        "Total Return",
        "Annualized Return",
        "Annualized Volatility",
        "Max Drawdown",
        "Value at Risk (95%)",
        "Conditional VaR (95%)",
    ]

    # Create a new DataFrame for the formatted values
    formatted_metrics_display = pd.DataFrame(
        index=formatted_metrics.index, columns=formatted_metrics.columns
    )

    # Copy values, formatting percentages where appropriate
    for idx in formatted_metrics.index:
        for col in formatted_metrics.columns:
            if idx in percentage_rows:
                formatted_metrics_display.loc[
                    idx, col
                ] = f"{formatted_metrics.loc[idx, col]:.2%}"
            else:
                formatted_metrics_display.loc[idx, col] = formatted_metrics.loc[
                    idx, col
                ]

    # Convert to Markdown
    metrics_table = formatted_metrics.transpose().to_markdown()
    report += metrics_table + "\n\n"

    # Add correlation information
    report += "## Return Correlations\n\n"
    correlation = returns.corr()
    correlation_table = correlation.to_markdown()
    report += correlation_table + "\n\n"

    # Add benchmark descriptions
    report += "## Benchmark Descriptions\n\n"

    for column in returns.columns:
        report += f"### {column}\n\n"

        # Default descriptions based on common benchmark names
        if column.lower() == "spy" or column.lower() == "s&p 500":
            report += "The S&P 500 Index is a market-capitalization-weighted index of the 500 largest publicly traded companies in the U.S. It is widely regarded as the best gauge of large-cap U.S. equities.\n\n"
        elif "60/40" in column.lower() or "balanced" in column.lower():
            report += "A traditional balanced portfolio consisting of 60% stocks (S&P 500) and 40% bonds. This allocation is considered a benchmark for moderate investors seeking growth with some downside protection.\n\n"
        elif "fa" in column.lower() or "fast algorithm" in column.lower():
            report += "A portfolio optimized using the Fast Algorithm implementation of the Markowitz Mean-Variance Optimization framework. This portfolio seeks to maximize the Sharpe ratio by finding the optimal allocation across assets.\n\n"
        else:
            report += "Custom benchmark or strategy.\n\n"

    # Add notes
    report += "## Notes\n\n"
    report += "- Sharpe Ratio assumes a risk-free rate of 0%.\n"
    report += "- Max Drawdown represents the largest peak-to-trough decline during the period.\n"
    report += "- Value at Risk (95%) indicates the worst expected loss over a day with 95% confidence.\n"

    # Save if output_file is provided
    if output_file:
        with open(output_file, "w") as f:
            f.write(report)
        print(f"Saved benchmark analysis report to {output_file}")

    return report


def main(
    return_files: List[str],
    names: Optional[List[str]] = None,
    risk_free_rate: float = 0.0,
    output_dir: Optional[str] = None,
) -> None:
    """
    Main function to analyze benchmark returns.

    Parameters:
    -----------
    return_files : List[str]
        List of paths to CSV files containing return data
    names : Optional[List[str]], optional
        Names to use for each return series, by default None
    risk_free_rate : float, optional
        Annualized risk-free rate, by default 0.0
    output_dir : Optional[str], optional
        Directory to save the output files, by default None
    """
    print("Analyzing benchmark returns")

    # Create output directory if specified and doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load return data
    returns = load_return_files(return_files, names)

    # Calculate performance metrics
    metrics = calculate_performance_metrics(returns, risk_free_rate)

    # Display metrics
    print("\nPerformance Metrics:")
    print(metrics)

    # Save metrics to CSV if output_dir is provided
    if output_dir:
        metrics_file = os.path.join(output_dir, "benchmark_metrics.csv")
        metrics.to_csv(metrics_file)
        print(f"Saved performance metrics to {metrics_file}")

    # Plot cumulative returns
    if output_dir:
        cumulative_file = os.path.join(output_dir, "cumulative_returns.png")
        plot_cumulative_returns(returns, cumulative_file)
    else:
        plot_cumulative_returns(returns)

    # Plot rolling metrics
    plot_rolling_metrics(returns, window=252, output_dir=output_dir)

    # Analyze drawdowns
    try:
        if output_dir:
            drawdown_file = os.path.join(output_dir, "drawdowns.png")
            drawdown_stats, _ = analyze_drawdowns(returns, drawdown_file)
        else:
            drawdown_stats, _ = analyze_drawdowns(returns)

        # Print drawdown statistics
        print("\nDrawdown Statistics:")
        # Configure pandas to show all rows and columns
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)  # Wider display to prevent line wrapping
        print(drawdown_stats)
    except Exception as e:
        print(f"Warning: Error during drawdown analysis: {e}")
        drawdown_stats = pd.DataFrame()

    # Pandas display settings were already configured in the try block

    # Create summary report
    if output_dir:
        report_file = os.path.join(output_dir, "benchmark_analysis.md")
        create_summary_report(returns, metrics, report_file)
    else:
        report = create_summary_report(returns, metrics)
        print("\nSummary Report:")
        print(report)

    print("\nBenchmark analysis complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze benchmark returns")
    parser.add_argument(
        "--files",
        "-f",
        nargs="+",
        required=True,
        help="Paths to CSV files containing return data",
    )
    parser.add_argument(
        "--names",
        "-n",
        nargs="+",
        help="Names for each return series (must match number of files)",
    )
    parser.add_argument(
        "--risk-free-rate",
        "-r",
        type=float,
        default=0.0,
        help="Annualized risk-free rate",
    )
    parser.add_argument("--output", "-o", help="Directory to save output files")

    args = parser.parse_args()

    if args.names and len(args.names) != len(args.files):
        print(
            "Warning: Number of names does not match number of files. Using default names."
        )
        args.names = None

    main(
        args.files,
        names=args.names,
        risk_free_rate=args.risk_free_rate,
        output_dir=args.output,
    )
