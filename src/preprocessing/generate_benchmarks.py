"""
Enhanced Benchmark Generation Module

This script generates benchmark returns with customizable rebalancing options:
1. S&P 500 (SPY) returns
2. 60/40 Stock/Bond portfolio returns with rebalancing
"""

import argparse
import os
from typing import Optional

import pandas as pd

from src.config.periods import get_period_range


def load_returns_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load ETF returns data from CSV file.

    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing returns data

    Returns:
    --------
    Optional[pd.DataFrame]
        Returns data or None if an error occurs
    """
    try:
        returns = pd.read_csv(file_path, index_col=0, parse_dates=True)
        print(f"Successfully loaded returns data from {file_path}")
        print(f"Shape: {returns.shape}")
        return returns
    except Exception as e:
        print(f"Error loading returns data: {e}")
        return None


def create_spy_benchmark(returns_data: pd.DataFrame) -> pd.Series:
    """
    Extract SPY returns as a benchmark.

    Parameters:
    -----------
    returns_data : pd.DataFrame
        Returns data for multiple ETFs including SPY

    Returns:
    --------
    pd.Series
        SPY returns
    """
    if "SPY" in returns_data.columns:
        spy_returns = returns_data["SPY"]
        print(f"Created SPY benchmark with {len(spy_returns)} data points")
        return spy_returns
    else:
        raise ValueError("SPY data not found in the returns dataset")


def create_balanced_portfolio(
    returns_data: pd.DataFrame,
    stock_symbol: str = "SPY",
    bond_symbol: str = "BND",
    stock_weight: float = 0.6,
) -> pd.Series:
    """
    Create a balanced portfolio with stocks and bonds (default: 60/40) without rebalancing.

    Parameters:
    -----------
    returns_data : pd.DataFrame
        Returns data for multiple ETFs
    stock_symbol : str, optional
        Symbol for the stock ETF, by default 'SPY'
    bond_symbol : str, optional
        Symbol for the bond ETF, by default 'BND'
    stock_weight : float, optional
        Weight for stocks in the portfolio, by default 0.6

    Returns:
    --------
    pd.Series
        Returns for the balanced portfolio without rebalancing
    """
    # Ensure both symbols exist in the dataset
    if stock_symbol not in returns_data.columns:
        raise ValueError(f"Stock symbol {stock_symbol} not found in dataset")

    if bond_symbol not in returns_data.columns:
        raise ValueError(f"Bond symbol {bond_symbol} not found in dataset")

    # Extract the returns
    stock_returns = returns_data[stock_symbol]
    bond_returns = returns_data[bond_symbol]

    # Calculate weighted returns (without rebalancing)
    balanced_returns = stock_returns * stock_weight + bond_returns * (1 - stock_weight)
    balanced_returns.name = f"{stock_symbol}/{bond_symbol} No Rebalancing"

    print(
        f"Created {stock_weight * 100:.0f}/{(1 - stock_weight) * 100:.0f} "
        f"{stock_symbol}/{bond_symbol} portfolio (no rebalancing) with {len(balanced_returns)} data points"
    )

    return balanced_returns


def create_periodically_rebalanced_portfolio(
    returns_data: pd.DataFrame,
    stock_symbol: str = "SPY",
    bond_symbol: str = "BND",
    stock_weight: float = 0.6,
    rebalance_freq: str = "M",
) -> pd.Series:
    """
    Create a balanced portfolio with periodic rebalancing.

    Parameters:
    -----------
    returns_data : pd.DataFrame
        Returns data for multiple ETFs
    stock_symbol : str, optional
        Symbol for the stock ETF, by default 'SPY'
    bond_symbol : str, optional
        Symbol for the bond ETF, by default 'BND'
    stock_weight : float, optional
        Weight for stocks in the portfolio, by default 0.6
    rebalance_freq : str, optional
        Rebalancing frequency: 'D' (daily), 'W' (weekly), 'M' (monthly),
        'Q' (quarterly), 'A' (annual), by default 'M'

    Returns:
    --------
    pd.Series
        Returns for the periodically rebalanced balanced portfolio
    """
    # Ensure both symbols exist in the dataset
    if stock_symbol not in returns_data.columns:
        raise ValueError(f"Stock symbol {stock_symbol} not found in dataset")

    if bond_symbol not in returns_data.columns:
        raise ValueError(f"Bond symbol {bond_symbol} not found in dataset")

    # Extract returns and ensure they're aligned
    combined = pd.DataFrame(
        {"stock": returns_data[stock_symbol], "bond": returns_data[bond_symbol]}
    ).dropna()

    # If daily rebalancing is selected, it's equivalent to weighted daily returns
    if rebalance_freq == "D":
        rebalanced_returns = (
            stock_weight * combined["stock"] + (1 - stock_weight) * combined["bond"]
        )
        rebalancing_type = "Daily"
    else:
        # For other frequencies, we need to track portfolio value with rebalancing
        # Start with $1 in each asset
        portfolio_value = 1.0
        # rebalanced_values = []

        # Get the rebalancing dates based on the chosen frequency
        rebal_dates = pd.date_range(
            start=combined.index.min(), end=combined.index.max(), freq=rebalance_freq
        )

        # Add the first date if it's not already included
        if combined.index.min() not in rebal_dates:
            rebal_dates = rebal_dates.insert(0, combined.index.min())

        # Create daily values series
        daily_values = pd.Series(index=combined.index)
        daily_values.iloc[0] = portfolio_value

        last_rebal_date = combined.index[0]
        stock_alloc = stock_weight * portfolio_value
        bond_alloc = (1 - stock_weight) * portfolio_value

        # Calculate daily portfolio values
        for date in combined.index[1:]:
            # Get daily returns
            stock_return = combined.loc[date, "stock"]
            bond_return = combined.loc[date, "bond"]

            # Update allocations based on the day's returns
            stock_alloc *= 1 + stock_return
            bond_alloc *= 1 + bond_return

            # Calculate new portfolio value
            new_portfolio_value = stock_alloc + bond_alloc

            # Store the value
            daily_values[date] = new_portfolio_value

            # Check if this is a rebalancing date
            if date in rebal_dates:
                # Rebalance
                stock_alloc = stock_weight * new_portfolio_value
                bond_alloc = (1 - stock_weight) * new_portfolio_value
                # last_rebal_date = date

        # Calculate daily returns from the portfolio values
        rebalanced_returns = daily_values.pct_change().dropna()

        # Map frequency codes to description
        freq_names = {"W": "Weekly", "M": "Monthly", "Q": "Quarterly", "A": "Annual"}
        rebalancing_type = freq_names.get(rebalance_freq, rebalance_freq)

    # Set name for the series
    rebalanced_returns.name = (
        f"{stock_symbol}/{bond_symbol} {rebalancing_type} Rebalancing"
    )

    print(
        f"Created {stock_weight * 100:.0f}/{(1 - stock_weight) * 100:.0f} "
        f"{stock_symbol}/{bond_symbol} portfolio ({rebalancing_type.lower()} rebalancing) "
        f"with {len(rebalanced_returns)} data points"
    )

    return rebalanced_returns


def create_threshold_rebalanced_portfolio(
    returns_data: pd.DataFrame,
    stock_symbol: str = "SPY",
    bond_symbol: str = "BND",
    stock_weight: float = 0.6,
    threshold: float = 0.05,
) -> pd.Series:
    """
    Create a balanced portfolio with threshold-based rebalancing.

    Parameters:
    -----------
    returns_data : pd.DataFrame
        Returns data for multiple ETFs
    stock_symbol : str, optional
        Symbol for the stock ETF, by default 'SPY'
    bond_symbol : str, optional
        Symbol for the bond ETF, by default 'BND'
    stock_weight : float, optional
        Target weight for stocks in the portfolio, by default 0.6
    threshold : float, optional
        Rebalancing threshold (% deviation from target), by default 0.05

    Returns:
    --------
    pd.Series
        Returns for the threshold-rebalanced balanced portfolio
    """
    # Ensure both symbols exist in the dataset
    if stock_symbol not in returns_data.columns:
        raise ValueError(f"Stock symbol {stock_symbol} not found in dataset")

    if bond_symbol not in returns_data.columns:
        raise ValueError(f"Bond symbol {bond_symbol} not found in dataset")

    # Extract returns and ensure they're aligned
    combined = pd.DataFrame(
        {"stock": returns_data[stock_symbol], "bond": returns_data[bond_symbol]}
    ).dropna()

    # Start with $1 in each asset based on target weights
    portfolio_value = 1.0
    stock_alloc = stock_weight * portfolio_value
    bond_alloc = (1 - stock_weight) * portfolio_value

    # Create daily values series
    daily_values = pd.Series(index=combined.index)
    daily_values.iloc[0] = portfolio_value

    # Track rebalancing dates for reporting
    rebalancing_dates = []

    # Calculate daily portfolio values
    for date in combined.index[1:]:
        # Get daily returns
        stock_return = combined.loc[date, "stock"]
        bond_return = combined.loc[date, "bond"]

        # Update allocations based on the day's returns
        stock_alloc *= 1 + stock_return
        bond_alloc *= 1 + bond_return

        # Calculate new portfolio value
        new_portfolio_value = stock_alloc + bond_alloc

        # Calculate current stock weight
        current_stock_weight = stock_alloc / new_portfolio_value

        # Check if rebalancing is needed
        if abs(current_stock_weight - stock_weight) > threshold:
            # Rebalance
            stock_alloc = stock_weight * new_portfolio_value
            bond_alloc = (1 - stock_weight) * new_portfolio_value
            rebalancing_dates.append(date)

        # Store the value
        daily_values[date] = new_portfolio_value

    # Calculate daily returns from the portfolio values
    rebalanced_returns = daily_values.pct_change().dropna()

    # Set name for the series
    rebalanced_returns.name = (
        f"{stock_symbol}/{bond_symbol} {threshold * 100:.0f}% Threshold Rebalancing"
    )

    print(
        f"Created {stock_weight * 100:.0f}/{(1 - stock_weight) * 100:.0f} "
        f"{stock_symbol}/{bond_symbol} portfolio ({threshold * 100:.0f}% threshold rebalancing) "
        f"with {len(rebalanced_returns)} data points"
    )
    print(
        f"Rebalanced {len(rebalancing_dates)} times over {len(combined)} trading days"
    )

    return rebalanced_returns


def main(
    returns_file: str,
    output_dir: str,
    stock_symbol: str = "SPY",
    bond_symbol: str = "BND",
    stock_weight: float = 0.6,
    rebalance_method: str = "periodic",
    rebalance_freq: str = "M",
    rebalance_threshold: float = 0.05,
    period: str = None,
    start_date: str = None,
    end_date: str = None,
) -> None:
    """
    Main function to generate benchmark returns.

    Parameters:
    -----------
    returns_file : str
        Path to the CSV file containing returns data
    output_dir : str
        Directory to save the benchmark returns
    stock_symbol : str, optional
        Symbol for the stock ETF, by default 'SPY'
    bond_symbol : str, optional
        Symbol for the bond ETF, by default 'BND'
    stock_weight : float, optional
        Weight for stocks in the balanced portfolio, by default 0.6
    rebalance_method : str, optional
        Rebalancing method: 'none', 'periodic', 'threshold', by default 'periodic'
    rebalance_freq : str, optional
        Rebalancing frequency for periodic method, by default 'M'
    rebalance_threshold : float, optional
        Rebalancing threshold for threshold method, by default 0.05
    """
    print("Generating benchmark returns")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load returns data
    returns_data = load_returns_data(returns_file)
    if returns_data is None:
        return

    # Filter returns_data by period or start/end date if provided
    if period and period != "custom":
        if not start_date or not end_date:
            try:
                default_start, default_end = get_period_range(period)
                if not start_date:
                    start_date = default_start
                if not end_date:
                    end_date = default_end
            except Exception as e:
                print(f"Warning: Could not get period range for '{period}': {e}")
    if period or start_date or end_date:
        if start_date:
            returns_data = returns_data[
                returns_data.index >= pd.to_datetime(start_date)
            ]
        if end_date:
            returns_data = returns_data[returns_data.index <= pd.to_datetime(end_date)]
    # Print the start and end dates for the benchmark data
    if not returns_data.empty:
        print(
            f"Benchmark data start date: {returns_data.index[0].strftime('%Y-%m-%d')}"
        )
        print(f"Benchmark data end date: {returns_data.index[-1].strftime('%Y-%m-%d')}")
    else:
        print("Benchmark data is empty!")

    # Create SPY benchmark
    try:
        spy_returns = create_spy_benchmark(returns_data)
        spy_file = os.path.join(output_dir, "spy_returns.csv")
        spy_returns.to_frame("SPY").to_csv(spy_file)
        print(f"Saved SPY benchmark returns to {spy_file}")
    except Exception as e:
        print(f"Error creating SPY benchmark: {e}")

    # Create balanced portfolio benchmark based on selected rebalancing method
    try:
        if rebalance_method == "none":
            # No rebalancing
            balanced_returns = create_balanced_portfolio(
                returns_data,
                stock_symbol=stock_symbol,
                bond_symbol=bond_symbol,
                stock_weight=stock_weight,
            )
            balanced_name = "balanced_returns_no_rebal"
        elif rebalance_method == "threshold":
            # Threshold-based rebalancing
            balanced_returns = create_threshold_rebalanced_portfolio(
                returns_data,
                stock_symbol=stock_symbol,
                bond_symbol=bond_symbol,
                stock_weight=stock_weight,
                threshold=rebalance_threshold,
            )
            balanced_name = (
                f"balanced_returns_threshold_{int(rebalance_threshold * 100)}pct"
            )
        else:
            # Periodic rebalancing (default)
            balanced_returns = create_periodically_rebalanced_portfolio(
                returns_data,
                stock_symbol=stock_symbol,
                bond_symbol=bond_symbol,
                stock_weight=stock_weight,
                rebalance_freq=rebalance_freq,
            )
            balanced_name = f"balanced_returns_{rebalance_freq}_rebal"

        # Save to CSV
        balanced_file = os.path.join(output_dir, f"{balanced_name}.csv")
        balanced_returns.to_frame("60/40").to_csv(balanced_file)
        print(f"Saved balanced portfolio returns to {balanced_file}")

        # Also save a copy as the default balanced_returns.csv for compatibility
        default_balanced_file = os.path.join(output_dir, "balanced_returns.csv")
        balanced_returns.to_frame("60/40").to_csv(default_balanced_file)
        print(f"Saved copy as default balanced returns to {default_balanced_file}")

    except Exception as e:
        print(f"Error creating balanced portfolio benchmark: {e}")

    print("\nBenchmark generation complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate benchmark returns for portfolio comparison"
    )
    parser.add_argument(
        "--returns", "-r", required=True, help="Path to the returns CSV file"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Directory to save benchmark returns"
    )
    parser.add_argument("--stock", "-s", default="SPY", help="Stock ETF symbol")
    parser.add_argument("--bond", "-b", default="BND", help="Bond ETF symbol")
    parser.add_argument(
        "--weight",
        "-w",
        type=float,
        default=0.6,
        help="Weight for stocks in balanced portfolio",
    )
    parser.add_argument(
        "--period",
        "-p",
        type=str,
        default=None,
        help="Period name to filter data (e.g., recent, financial_crisis, post_crisis)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date to filter data (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date to filter data (YYYY-MM-DD)",
    )
    # Rebalancing options
    parser.add_argument(
        "--rebalance",
        choices=["none", "periodic", "threshold"],
        default="periodic",
        help="Rebalancing method",
    )
    parser.add_argument(
        "--frequency",
        choices=["D", "W", "M", "Q", "A"],
        default="M",
        help="Rebalancing frequency for periodic method",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Rebalancing threshold (as decimal) for threshold method",
    )

    args = parser.parse_args()

    main(
        args.returns,
        args.output,
        stock_symbol=args.stock,
        bond_symbol=args.bond,
        stock_weight=args.weight,
        rebalance_method=args.rebalance,
        rebalance_freq=args.frequency,
        rebalance_threshold=args.threshold,
        period=args.period,
        start_date=args.start_date,
        end_date=args.end_date,
    )
