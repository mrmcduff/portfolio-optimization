"""
Benchmark Generation Module

This script generates benchmark returns for performance comparison:
1. S&P 500 (SPY) returns
2. 60/40 Stock/Bond portfolio returns
"""

import argparse
import os
from typing import Optional

import pandas as pd


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
    Create a balanced portfolio with stocks and bonds (default: 60/40).

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
        Returns for the balanced portfolio
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

    print(
        f"Created {stock_weight*100:.0f}/{(1-stock_weight)*100:.0f} "
        f"{stock_symbol}/{bond_symbol} portfolio with {len(balanced_returns)} data points"
    )

    return balanced_returns


def main(
    returns_file: str,
    output_dir: str,
    stock_symbol: str = "SPY",
    bond_symbol: str = "BND",
    stock_weight: float = 0.6,
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
    """
    print("Generating benchmark returns")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load returns data
    returns_data = load_returns_data(returns_file)
    if returns_data is None:
        return

    # Create SPY benchmark
    try:
        spy_returns = create_spy_benchmark(returns_data)
        spy_file = os.path.join(output_dir, "spy_returns.csv")
        spy_returns.to_frame("SPY").to_csv(spy_file)
        print(f"Saved SPY benchmark returns to {spy_file}")
    except Exception as e:
        print(f"Error creating SPY benchmark: {e}")

    # Create balanced portfolio benchmark
    try:
        balanced_returns = create_balanced_portfolio(
            returns_data,
            stock_symbol=stock_symbol,
            bond_symbol=bond_symbol,
            stock_weight=stock_weight,
        )

        balanced_file = os.path.join(output_dir, "balanced_returns.csv")
        balanced_returns.to_frame("60/40").to_csv(balanced_file)
        print(f"Saved balanced portfolio returns to {balanced_file}")
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

    args = parser.parse_args()

    main(
        args.returns,
        args.output,
        stock_symbol=args.stock,
        bond_symbol=args.bond,
        stock_weight=args.weight,
    )
