"""
Benchmark Analysis Module

This script analyzes benchmark returns and generates a summary report
of key performance metrics for comparison with optimized portfolios.
"""

import argparse
import ast
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_one_factor_fast_composition(output_dir: str):
    """
    Load one_factor_fast_rebalancing.xlsx and plot a normalized stacked area chart
    showing the portfolio composition (weights) over time.
    """
    import os

    rebal_path = os.path.join("results/models", "one_factor_fast_rebalancing.xlsx")
    if not os.path.exists(rebal_path):
        print(f"Rebalancing file not found: {rebal_path}")
        return
    df = pd.read_excel(rebal_path)
    # Filter by start_date and end_date using 'end_date' column only (for consistency)
    start_date_env = os.environ.get("BENCHMARK_START_DATE")
    end_date_env = os.environ.get("BENCHMARK_END_DATE")
    if start_date_env:
        df = df[pd.to_datetime(df["end_date"]) >= pd.to_datetime(start_date_env)]
    if end_date_env:
        df = df[pd.to_datetime(df["end_date"]) <= pd.to_datetime(end_date_env)]

    # Parse weights (stored as dicts in Excel)
    if isinstance(df.loc[0, "weights"], str):
        df["weights"] = df["weights"].apply(ast.literal_eval)

    # Build a DataFrame: index=rebalancing date, columns=securities, values=weights
    weight_records = []
    for idx, row in df.iterrows():
        date = pd.to_datetime(row["end_date"])  # Use end date for time axis
        wdict = row["weights"]
        for security, weight in wdict.items():
            weight_records.append(
                {"date": date, "security": security, "weight": weight}
            )
    weights_long = pd.DataFrame(weight_records)
    weights_pivot = weights_long.pivot(
        index="date", columns="security", values="weight"
    ).fillna(0)
    # Normalize (should sum to 1, but enforce)
    weights_norm = weights_pivot.div(weights_pivot.sum(axis=1), axis=0)

    # Plot
    plt.figure(figsize=(14, 7))
    if (weights_norm < 0).any().any():
        print(
            "Negative weights detected: plotting as line chart (not stacked area) due to short sales."
        )
        weights_norm.plot(ax=plt.gca(), cmap="tab20")
    else:
        weights_norm.plot.area(ax=plt.gca(), stacked=True, cmap="tab20")
    plt.title("One-Factor Fast Algorithm Portfolio Composition Over Time")
    plt.ylabel("Portfolio Weight")
    plt.xlabel("Date")
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), title="Security")
    plt.tight_layout()
    outpath = os.path.join(output_dir, "one_factor_fast_composition.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved composition plot to {outpath}")


def plot_custom_algorithm_composition(output_dir: str, period: str = None):
    """
    Load custom_algorithm_rebalancing.xlsx and plot a normalized stacked area chart
    showing the portfolio composition (weights) over time.
    Also checks for long/short versions and plots them as well.
    """
    import os

    # Check for different versions of rebalancing files
    base_rebal_path = os.path.join(
        "results/models", "custom_algorithm_rebalancing.xlsx"
    )
    long_rebal_path = os.path.join(
        "results/models", "custom_long_algorithm_rebalancing.xlsx"
    )
    short_rebal_path = os.path.join(
        "results/models", "custom_short_algorithm_rebalancing.xlsx"
    )

    rebal_paths = []

    # Determine which files exist and should be plotted
    if os.path.exists(long_rebal_path) and os.path.exists(short_rebal_path):
        # Both long and short versions exist
        rebal_paths = [
            (
                long_rebal_path,
                "Custom Algorithm (Long) Portfolio Composition Over Time",
                "custom_algorithm_long_composition.png",
            ),
            (
                short_rebal_path,
                "Custom Algorithm (Short) Portfolio Composition Over Time",
                "custom_algorithm_short_composition.png",
            ),
        ]
    elif os.path.exists(base_rebal_path):
        # Just the standard version exists
        rebal_paths = [
            (
                base_rebal_path,
                "Custom Algorithm Portfolio Composition Over Time",
                "custom_algorithm_composition.png",
            )
        ]
    else:
        print("No rebalancing files found for custom algorithm")
        return

    # Plot each version
    for rebal_path, title, output_filename in rebal_paths:
        print(f"Processing rebalancing file: {rebal_path}")

        df = pd.read_excel(rebal_path)
        # Filter by start_date and end_date using date column
        start_date_env = os.environ.get("BENCHMARK_START_DATE")
        end_date_env = os.environ.get("BENCHMARK_END_DATE")

        # First check for 'end_date' column, fall back to 'date'
        date_col = "end_date" if "end_date" in df.columns else "date"

        if start_date_env:
            df = df[pd.to_datetime(df[date_col]) >= pd.to_datetime(start_date_env)]
        if end_date_env:
            df = df[pd.to_datetime(df[date_col]) <= pd.to_datetime(end_date_env)]

        # Parse weights (stored as dicts in Excel)
        if not df.empty and isinstance(df.loc[0, "weights"], str):
            df["weights"] = df["weights"].apply(ast.literal_eval)

        # Check if we have data
        if df.empty:
            print(f"No data in rebalancing file {rebal_path} after date filtering")
            continue

        # Build a DataFrame: index=rebalancing date, columns=securities, values=weights
        weight_records = []
        for idx, row in df.iterrows():
            date = pd.to_datetime(row["date"])  # Use rebalancing date for time axis
            wdict = row["weights"]
            for security, weight in wdict.items():
                weight_records.append(
                    {"date": date, "security": security, "weight": weight}
                )
        weights_long = pd.DataFrame(weight_records)

        if weights_long.empty:
            print(f"No weight records found in {rebal_path}")
            continue

        weights_pivot = weights_long.pivot(
            index="date", columns="security", values="weight"
        ).fillna(0)

        # Normalize (should sum to 1, but enforce)
        weights_norm = weights_pivot.div(weights_pivot.sum(axis=1), axis=0)

        # Plot
        plt.figure(figsize=(14, 7))
        has_negative = (weights_norm < 0).any().any()

        if has_negative:
            print(
                f"Negative weights detected in {rebal_path}: plotting as line chart (not stacked area) due to short sales."
            )
            ax = plt.gca()
            weights_norm.plot(ax=ax, cmap="tab20")

            # Add a horizontal line at y=0 to clearly show the boundary between long and short positions
            ax.axhline(y=0, color="black", linestyle="-", linewidth=1.0, alpha=0.7)

            # Add grid lines to make it easier to see position sizes
            ax.grid(True, linestyle="--", alpha=0.6)
        else:
            weights_norm.plot.area(ax=plt.gca(), stacked=True, cmap="tab20")

        plt.title(title)
        plt.ylabel("Portfolio Weight")
        plt.xlabel("Date")
        plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), title="Security")
        plt.tight_layout()
        outpath = os.path.join(output_dir, output_filename)
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved composition plot to {outpath}")


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

        # Calculate common date range before alignment (for logging)
        orig_first_date = combined.index.min()
        orig_last_date = combined.index.max()
        orig_count = len(combined)

        # Look for known period boundaries
        # Define standard periods for analysis
        periods = {
            "financial_crisis": ("2008-01-02", "2013-01-02"),
            "post_crisis": ("2012-01-01", "2018-12-31"),
            "recent": ("2019-01-01", "2023-12-31"),
        }
        period_of_interest = "recent"  # Default to recent

        # Check if any environment variables are set to override this
        import os

        period_env = os.environ.get("BENCHMARK_PERIOD")
        if period_env and period_env in periods:
            period_of_interest = period_env

        # Get the date range for the period of interest
        period_start = pd.to_datetime(periods[period_of_interest][0])
        period_end = pd.to_datetime(periods[period_of_interest][1])

        # Identify which series are missing data within the period of interest
        period_mask = (combined.index >= period_start) & (combined.index <= period_end)
        period_data = combined.loc[period_mask]

        # Check each series for NaN values in the period of interest
        nan_counts = {}
        missing_ranges = {}
        for col in period_data.columns:
            nan_series = period_data[col].isna()
            count = nan_series.sum()
            if count > 0:
                nan_counts[col] = count
                # Find first and last date with missing data
                missing_dates = period_data.index[nan_series]
                if len(missing_dates) > 0:
                    # Group consecutive dates - this is simplified logic
                    ranges = []
                    current_range = [missing_dates[0]]
                    for i in range(1, len(missing_dates)):
                        if (
                            missing_dates[i] - missing_dates[i - 1]
                        ).days > 5:  # New range if gap > 5 days
                            ranges.append((current_range[0], current_range[-1]))
                            current_range = [missing_dates[i]]
                        else:
                            current_range.append(missing_dates[i])
                    ranges.append((current_range[0], current_range[-1]))
                    missing_ranges[col] = ranges

        # Trim to common date range to ensure fair comparison
        # Drop NaN values to find valid range across all series
        combined = combined.dropna()

        if len(combined) < orig_count:
            first_valid_date = combined.index.min()
            last_valid_date = combined.index.max()
            print("Aligned all returns to common date range:")
            print(
                f"  Original: {orig_first_date.strftime('%Y-%m-%d')} to {orig_last_date.strftime('%Y-%m-%d')} ({orig_count} trading days)"
            )
            print(
                f"  Aligned: {first_valid_date.strftime('%Y-%m-%d')} to {last_valid_date.strftime('%Y-%m-%d')} ({len(combined)} trading days)"
            )
            print(
                f"  Dropped {orig_count - len(combined)} days with missing data in at least one series"
            )

            # Report which series were missing data in the period of interest
            if nan_counts:
                print(
                    f"\nSeries with missing data in {period_of_interest} period ({period_start.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}):"
                )
                for col, count in nan_counts.items():
                    percent = count / len(period_data) * 100
                    print(f"  {col}: {count} missing values ({percent:.2f}% of period)")

                    # Show detailed ranges of missing data
                    if col in missing_ranges:
                        print("    Missing data in these date ranges:")
                        for range_start, range_end in missing_ranges[col]:
                            days = (range_end - range_start).days + 1
                            print(
                                f"      {range_start.strftime('%Y-%m-%d')} to {range_end.strftime('%Y-%m-%d')} ({days} days)"
                            )
            else:
                print(
                    f"\nNo series missing data within the {period_of_interest} period."
                )

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
    daily_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

    # Identify market benchmark (assuming first column is the market)
    market_returns = returns.iloc[:, 0]
    market_excess_returns = market_returns - daily_rf

    for column in returns.columns:
        # Align the series with market returns to ensure same length
        series = returns[column].dropna()
        aligned_market = market_returns[series.index]
        aligned_market_excess = market_excess_returns[series.index]
        excess_returns = series - daily_rf

        # Calculate metrics
        total_return = (1 + series).prod() - 1
        ann_return = (1 + series.mean()) ** periods_per_year - 1
        ann_volatility = series.std() * np.sqrt(periods_per_year)
        sharpe_ratio = (
            (ann_return - risk_free_rate) / ann_volatility if ann_volatility > 0 else 0
        )

        # Calculate beta and Jensen's Alpha
        covariance = np.cov(excess_returns, aligned_market_excess)[0, 1]
        market_variance = np.var(aligned_market_excess)
        beta = covariance / market_variance if market_variance > 0 else 1.0

        # Calculate Jensen's Alpha (annualized)
        expected_return = daily_rf + beta * (aligned_market - daily_rf)
        alpha = (1 + (series - expected_return).mean()) ** periods_per_year - 1

        # Maximum drawdown
        cumulative_returns = (1 + series).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        max_drawdown = drawdown.min()

        # Value at Risk (95%) and Conditional VaR
        var_95 = series.quantile(0.05)
        cvar_95 = series[series <= var_95].mean()

        # Skewness and kurtosis
        skewness = series.skew()
        kurtosis = series.kurtosis()

        # Store metrics
        metrics[column] = {
            "Total Return": total_return,
            "Annualized Return": ann_return,
            "Annualized Volatility": ann_volatility,
            "Sharpe Ratio": sharpe_ratio,
            "Jensen's Alpha": alpha,
            "Beta": beta,
            "Max Drawdown": max_drawdown,
            "Value at Risk (95%)": var_95,
            "Conditional VaR (95%)": cvar_95,
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
    rolling_returns = returns.rolling(window=window, min_periods=1).mean() * window
    rolling_vol = returns.rolling(window=window, min_periods=1).std() * np.sqrt(window)
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


# --- HISTOGRAM FUNCTION (must be defined before CLI block) ---
def plot_histograms_of_annualized_returns(verbose: bool = False):
    """
    Plot histograms of annualized returns per week for SPY, balanced, custom, and fast algorithms.
    Save to results/figures as histogram_spy.png, histogram_balanced.png, histogram_custom.png, histogram_fast.png.
    If both long-only and short-enabled custom algorithm versions exist, plot both.
    Print the file path used for each graph, and print mean/median annualized weekly return values.
    """
    import os

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    FIGURES_DIR = "results/figures"
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Check for both long and short custom algorithm versions
    cust_long_path = "results/models/cust_long_portfolio_returns.csv"
    cust_short_path = "results/models/cust_short_portfolio_returns.csv"
    cust_path = "results/models/cust_portfolio_returns.csv"

    files = {
        "SPY": "results/models/spy_returns.csv",
        "Balanced": "results/models/balanced_returns_M.csv",
        "Fast": "results/models/ofa_portfolio_returns.csv",
    }

    outnames = {
        "SPY": "histogram_spy.png",
        "Balanced": "histogram_balanced.png",
        "Fast": "histogram_fast.png",
    }

    # Add appropriate custom algorithm versions
    if os.path.exists(cust_long_path) and os.path.exists(cust_short_path):
        files["Custom (Long)"] = cust_long_path
        files["Custom (Short)"] = cust_short_path
        outnames["Custom (Long)"] = "histogram_custom_long.png"
        outnames["Custom (Short)"] = "histogram_custom_short.png"
    elif os.path.exists(cust_path):
        files["Custom"] = cust_path
        outnames["Custom"] = "histogram_custom.png"
    for label, fpath in files.items():
        print(f"Loading data for {label} from: {fpath}")
        import os

        base, ext = os.path.splitext(fpath)
        df = pd.read_csv(fpath, index_col=0, parse_dates=True)
        # Use first column if multi-column
        if df.shape[1] > 1:
            series = df.iloc[:, 0]
        else:
            series = df.squeeze()
        trailing_windows = {"daily": 252, "weekly": 52, "monthly": 12}
        resample_rules = {"daily": None, "weekly": "W", "monthly": "ME"}
        for freq, window in trailing_windows.items():
            if freq == "daily":
                returns = series
            else:
                returns = (1 + series).resample(resample_rules[freq]).prod() - 1
            trailing = (1 + returns).rolling(window).apply(np.prod, raw=True) - 1
            trailing = trailing.dropna()
            # Filter: Only include trailing returns after 1 year past the start of the test period
            if not trailing.empty:
                trailing_start_date = trailing.index[0] + pd.DateOffset(years=1)
                trailing = trailing[trailing.index >= trailing_start_date]
            # Save trailing returns to CSV
            trailing_csv = base + f"_trailing_1yr_{freq}.csv"
            trailing.to_csv(trailing_csv)
            print(f"Saved trailing 1-year {freq} returns to {trailing_csv}")
            # Print number of points to be plotted if verbose
            if verbose:
                print(f"Plotting {label} ({freq}): {len(trailing)} data points.")
            # Plot histogram
            plt.figure(figsize=(8, 5))
            plt.hist(
                trailing,
                bins=20,
                color="#1f77b4",
                alpha=0.7,
                label="Trailing 1Y Returns",
            )
            mean_val = trailing.mean()
            median_val = trailing.median()
            plt.axvline(
                mean_val,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {mean_val:.2%}",
            )
            plt.axvline(
                median_val,
                color="green",
                linestyle=":",
                linewidth=2,
                label=f"Median: {median_val:.2%}",
            )
            plt.title(f"Histogram of 1-Year Trailing Returns ({freq.title()}): {label}")
            plt.xlabel("1-Year Trailing Return")
            plt.ylabel("Frequency")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            fig_out = os.path.join(
                FIGURES_DIR, f"histogram_{label.lower()}_trailing_1yr_{freq}.png"
            )
            plt.savefig(fig_out, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved histogram to {fig_out}")

        outpath = os.path.join(FIGURES_DIR, outnames[label])
        plt.tight_layout()
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved histogram to {outpath}")


def shorten_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Shortens column names for better display in fixed-width tables.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns to shorten

    Returns:
    --------
    pd.DataFrame
        DataFrame with shortened column names
    """
    # Make a copy to avoid modifying the original
    result = df.copy()

    # Define mapping of full names to shortened versions
    short_names = {
        "S&P 500": "S&P 500",
        "60/40 Portfolio": "60/40",
        "Custom Algorithm": "Custom Algo",
        "Custom Algorithm (Long)": "Custom (LO)",
        "Custom Algorithm (Short)": "Custom (SH)",
        "One Factor Fast Algorithm": "Fast Algo",
        "Returns Algorithm": "Returns Algo",
        "Weighted Top Five": "WTF",
    }

    # Apply the mapping to columns
    result.columns = [short_names.get(col, col) for col in result.columns]

    # If it's a correlation matrix, also apply to index
    if isinstance(result.index, pd.Index) and any(
        idx in short_names for idx in result.index
    ):
        result.index = [short_names.get(idx, idx) for idx in result.index]

    return result


def format_table(data: pd.DataFrame, title: str) -> str:
    """
    Format a DataFrame as a markdown table with consistent column widths.

    Parameters:
    -----------
    data : pd.DataFrame
        Data to format as a table
    title : str
        Title for the table

    Returns:
    --------
    str
        Formatted markdown table
    """
    # Create a new DataFrame with string dtype for formatted values
    formatted_data = pd.DataFrame(index=data.index, columns=data.columns, dtype=str)

    # Define which metrics should be formatted as percentages
    percentage_metrics = [
        "Total Return",
        "Annualized Return",
        "Annualized Volatility",
        "Sharpe Ratio",
        "Jensen's Alpha",
        "Beta",
        "Max Drawdown",
        "Value at Risk (95%)",
        "Conditional VaR (95%)",
        "Skewness",
    ]

    for col in data.columns:
        for idx in data.index:
            value = data.loc[idx, col]
            if idx in percentage_metrics:
                # Format as percentage with 2 decimal places
                formatted_data.loc[idx, col] = f"{value:.2%}"
            elif idx == "Kurtosis":
                # Format kurtosis as float with 2 decimal places
                formatted_data.loc[idx, col] = f"{value:.2f}"
            else:
                # Format other metrics as float with 2 decimal places
                formatted_data.loc[idx, col] = f"{value:.2f}"

    # Create markdown table
    table = f"## {title}\n\n"

    # Apply column name shortening
    formatted_data = shorten_column_names(formatted_data)

    # Header row with wider metric column
    headers = ["Metric"] + list(formatted_data.columns)
    table += (
        "| "
        + " | ".join(
            f"{h:<21}" if i == 0 else f"{h:<12}" for i, h in enumerate(headers)
        )
        + " |\n"
    )

    # Separator row
    table += (
        "|:"
        + ":|:".join("-" * 21 if i == 0 else "-" * 12 for i, _ in enumerate(headers))
        + ":|\n"
    )

    # Data rows
    for idx, row in formatted_data.iterrows():
        values = [str(idx)] + [str(v) for v in row]
        table += (
            "| "
            + " | ".join(
                f"{v:<21}" if i == 0 else f"{v:<12}" for i, v in enumerate(values)
            )
            + " |\n"
        )

    return table + "\n"


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
        "Jensen's Alpha",
    ]

    # Apply shortened column names
    formatted_metrics = shorten_column_names(formatted_metrics)

    # Create a new DataFrame for the formatted values
    formatted_metrics_display = pd.DataFrame(
        index=formatted_metrics.index, columns=formatted_metrics.columns
    )

    # Copy values, formatting percentages where appropriate
    for idx in formatted_metrics.index:
        for col in formatted_metrics.columns:
            value = formatted_metrics.loc[idx, col]
            if idx in percentage_rows:
                formatted_metrics_display.loc[idx, col] = f"{value:.2%}"
            elif idx == "Beta":
                formatted_metrics_display.loc[idx, col] = f"{value:.2f}"
            else:
                formatted_metrics_display.loc[idx, col] = f"{value:.3f}"

    # Convert to Markdown
    metrics_table = formatted_metrics_display.to_markdown()
    report += metrics_table + "\n\n"

    # Add correlation information
    report += "## Return Correlations\n\n"
    correlation = returns.corr()
    # Apply shortened names to correlation table
    correlation = shorten_column_names(correlation)
    correlation_table = correlation.to_markdown()
    report += correlation_table + "\n\n"

    # Add benchmark descriptions
    report += "## Benchmark Descriptions\n\n"

    # Use shortened column names for benchmark descriptions
    returns_short = shorten_column_names(pd.DataFrame(columns=returns.columns))
    returns_columns_short = returns_short.columns.tolist()

    for i, column in enumerate(returns_columns_short):
        report += f"### {column}\n\n"

        # Special descriptions for different portfolio types
        if i == 0:  # First column (market benchmark)
            report += (
                "Market benchmark used for calculating Beta and Jensen's Alpha.\n\n"
            )
        elif "Custom (Long)" in column:
            report += "Custom Algorithm portfolio with long-only constraints.\n\n"
        elif "Custom (Short)" in column:
            report += "Custom Algorithm portfolio allowing short positions.\n\n"
        elif "Custom Algo" in column:
            report += (
                "Custom Algorithm portfolio as specified in the run configuration.\n\n"
            )
        elif "Fast Algo" in column:
            report += (
                "One Factor Fast Algorithm portfolio using the Markowitz framework.\n\n"
            )
        else:
            report += "Portfolio or benchmark returns series.\n\n"

    # Add notes
    report += "## Notes\n\n"
    report += "- Sharpe Ratio assumes a risk-free rate of 0%.\n"
    report += "- Max Drawdown represents the largest peak-to-trough decline during the period.\n"
    report += "- Value at Risk (95%) indicates the worst expected loss over a day with 95% confidence.\n"
    report += "- Conditional VaR (95%) represents the average loss on days when losses exceed the 95% VaR.\n"
    report += "- Jensen's Alpha measures the portfolio's excess return relative to what would be predicted by CAPM.\n"
    report += "- Beta measures the portfolio's systematic risk relative to the market benchmark.\n"
    report += f"- Market benchmark used: {returns_columns_short[0]}\n"

    # Save if output_file is provided
    if output_file:
        with open(output_file, "w") as f:
            f.write(report)
        print(f"Saved benchmark analysis report to {output_file}")

    return report


def main(
    return_files: List[str],
    names: List[str],
    risk_free_rate: float = 0.02,
    output_dir: str = "results/analysis",
) -> None:
    """
    Run the benchmark analysis.

    Parameters:
    -----------
    return_files : List[str]
        List of paths to return files
    names : List[str]
        Names for each return series
    risk_free_rate : float, optional
        Risk-free rate (annualized), by default 0.02
    output_dir : str, optional
        Directory to save output files, by default "results/analysis"
    """
    # Load returns
    returns = load_return_files(return_files, names)

    # Calculate performance metrics
    metrics = calculate_performance_metrics(returns, risk_free_rate)

    # Calculate correlations
    correlations = returns.corr()

    # Format tables
    metrics_table = format_table(metrics, "Performance Metrics")
    corr_table = format_table(correlations, "Return Correlations")

    # Combine tables
    summary = metrics_table + corr_table

    # Save summary
    os.makedirs(output_dir, exist_ok=True)
    summary_file = os.path.join(output_dir, "benchmark_analysis_summary.txt")
    with open(summary_file, "w") as f:
        f.write(summary)

    # Print summary to console
    print("\n=== Benchmark Analysis Summary ===\n")
    print(summary)

    # Generate plots
    plot_cumulative_returns(returns, output_dir)
    plot_rolling_metrics(returns, window=252, output_dir=output_dir)
    drawdowns, drawdown_plot = analyze_drawdowns(
        returns, os.path.join(output_dir, "drawdowns.png")
    )

    # Save drawdown analysis
    drawdown_file = os.path.join(output_dir, "drawdown_analysis.csv")
    drawdowns.to_csv(drawdown_file)
    print(f"Drawdown analysis saved to {drawdown_file}")

    # Save metrics
    metrics_file = os.path.join(output_dir, "performance_metrics.csv")
    metrics.to_csv(metrics_file)
    print(f"Performance metrics saved to {metrics_file}")

    print("\nBenchmark analysis completed successfully!")

    # Plot normalized stacked area chart for fast algorithm composition
    plot_one_factor_fast_composition(output_dir)
    # Plot normalized stacked area chart for custom algorithm composition
    plot_custom_algorithm_composition(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmark analysis")
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="List of CSV files containing return data",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        help="Names to use for each return series (optional)",
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.02,
        help="Annualized risk-free rate",
    )
    parser.add_argument(
        "--output",
        help="Directory to save output files",
    )
    parser.add_argument(
        "--period",
        "-p",
        type=str,
        default=None,
        help="Period name to filter data (e.g., recent, financial_crisis, post_crisis)",
    )

    args = parser.parse_args()

    # Add default files if not provided
    if not args.files:
        # Check if we have both long-only and short-enabled versions
        cust_long_path = "results/models/cust_long_portfolio_returns.csv"
        cust_short_path = "results/models/cust_short_portfolio_returns.csv"

        # Base set of files
        base_files = [
            "results/models/spy_returns.csv",
            "results/models/balanced_returns_M.csv",
        ]

        # Default names
        base_names = [
            "S&P 500",
            "60/40",
        ]

        # Check for dual custom algorithm versions
        if os.path.exists(cust_long_path) and os.path.exists(cust_short_path):
            # Add both versions
            base_files.extend([cust_long_path, cust_short_path])
            base_names.extend(["Custom Algorithm (Long)", "Custom Algorithm (Short)"])
        else:
            # Just add the standard version
            base_files.append("results/models/cust_portfolio_returns.csv")
            base_names.append("Custom Algorithm")

        # Add the remaining files
        base_files.extend(
            [
                "results/models/one_factor_fast_algorithm_returns.csv",
                "results/models/returns_algorithm_portfolio.csv",
                "results/models/weighted_top_five_portfolio.csv",
            ]
        )

        base_names.extend(
            [
                "One Factor Fast Algorithm",
                "Returns Algorithm",
                "Weighted Top Five",
            ]
        )

        # Only include files that exist
        args.files = [f for f in base_files if os.path.exists(f)]
        args.names = [n for i, n in enumerate(base_names) if i < len(args.files)]

    # Add default names if not provided but files are
    elif args.files and not args.names:
        # Build names based on files
        args.names = []
        for file_path in args.files:
            basename = os.path.basename(file_path)
            if "spy_returns" in basename:
                args.names.append("S&P 500")
            elif "balanced_returns" in basename:
                args.names.append("60/40")
            elif "cust_long_portfolio" in basename:
                args.names.append("Custom Algorithm (Long)")
            elif "cust_short_portfolio" in basename:
                args.names.append("Custom Algorithm (Short)")
            elif "cust_portfolio" in basename:
                args.names.append("Custom Algorithm")
            elif "one_factor_fast" in basename:
                args.names.append("One Factor Fast Algorithm")
            elif "returns_algorithm" in basename:
                args.names.append("Returns Algorithm")
            elif "weighted_top_five" in basename:
                args.names.append("Weighted Top Five")
            else:
                args.names.append(f"Portfolio {len(args.names) + 1}")

    # Print out the file paths here
    print("Files used for graph generation:")
    for f in args.files:
        print(f"  {f}")

    # Pass the period argument to the plot_custom_algorithm_composition function
    # Ensure output_dir is set
    output_dir = args.output or "results/analysis"
    main(args.files, args.names, args.risk_free_rate, output_dir)
    # After main, you may want to run the composition plot with the period
    # (Assuming output_dir is args.output or default)
    plot_custom_algorithm_composition(output_dir, args.period)

    # Run returns_sharpe_comparison.py to generate returns and Sharpe ratio graphs
    import subprocess
    import sys

    script_path = os.path.join(
        os.path.dirname(__file__), "returns_sharpe_comparison.py"
    )
    print("\nRunning returns_sharpe_comparison.py to generate comparison graphs...")
    subprocess.run([sys.executable, script_path], check=True)

    # Plot histograms of annualized returns per month for each algorithm
    plot_histograms_of_annualized_returns()


def plot_balanced_monthly_composition():
    """
    Plot a normalized stacked area chart of SPY and BND weights over time
    from the monthly rebalanced 60/40 portfolio (balanced_returns_M_rebal.csv).
    Output is saved to results/models/balanced_monthly.png.
    """
    import os

    import matplotlib.pyplot as plt
    import pandas as pd

    # Path to processed CSV
    inpath = os.path.join("results/models", "balanced_returns_M.csv")
    if not os.path.exists(inpath):
        print(f"Balanced returns file not found: {inpath}")
        return
    df = pd.read_csv(inpath, parse_dates=[0])
    # Accept either 'date' or 'Date' as column name
    date_col = "date" if "date" in df.columns else "Date"
    df = df.set_index(date_col)
    # Only keep the weight columns
    if not ("weight_spy" in df.columns and "weight_bnd" in df.columns):
        print("weight_spy and weight_bnd columns not found in the input CSV.")
        return
    weights = df[["weight_spy", "weight_bnd"]]
    # Normalize just in case
    weights_norm = weights.div(weights.sum(axis=1), axis=0)
    # Plot
    plt.figure(figsize=(14, 7))
    # Explicit colors: SPY (dark blue), BND (light blue)
    colors = ["#1f77b4", "#aec7e8"]  # matplotlib default blues
    weights_norm.plot.area(ax=plt.gca(), stacked=True, color=colors)
    plt.title("60/40 Portfolio (Monthly Rebalancing) Composition Over Time")
    plt.ylabel("Portfolio Weight")
    plt.xlabel("Date")
    # Legend patches must match area colors
    from matplotlib.patches import Patch

    legend_patches = [
        Patch(facecolor=colors[0], label="SPY"),
        Patch(facecolor=colors[1], label="BND"),
    ]
    plt.legend(
        handles=legend_patches,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        title="Asset",
    )
    plt.tight_layout()
    outpath = os.path.join("results/models", "balanced_monthly.png")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved composition plot to {outpath}")
