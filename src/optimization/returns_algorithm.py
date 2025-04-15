"""
Returns-Based Portfolio Selection Module

This module implements a portfolio selection strategy that:
1. Selects assets with highest returns over the prior 3-month period
2. Ensures no asset exceeds 20% of portfolio weight
3. Rebalances quarterly
"""

import argparse
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("returns_algorithm.log"),
    ],
)


def load_returns(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load returns data from CSV file.

    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing returns data

    Returns:
    --------
    Optional[pd.DataFrame]
        Returns data with dates as index or None if an error occurs
    """
    try:
        returns = pd.read_csv(file_path, index_col=0, parse_dates=True)
        logging.info(f"Successfully loaded returns data from {file_path}")
        logging.info(f"Shape: {returns.shape}")
        return returns
    except Exception as e:
        logging.error(f"Error loading returns data: {e}")
        return None


def calculate_rolling_returns(returns: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """
    Calculate rolling returns over the specified window.

    Parameters:
    -----------
    returns : pd.DataFrame
        Daily returns data
    window : int, optional
        Rolling window size in days, by default 63 (3 months)

    Returns:
    --------
    pd.DataFrame
        Rolling returns for each asset
    """
    # Calculate cumulative returns over the window
    rolling_returns = (1 + returns).rolling(window).apply(np.prod) - 1
    return rolling_returns


def select_top_performers(
    rolling_returns: pd.Series, max_assets: Optional[int] = None
) -> List[str]:
    """
    Select top performing assets based on rolling returns.

    Parameters:
    -----------
    rolling_returns : pd.Series
        Rolling returns for each asset
    max_assets : Optional[int], optional
        Maximum number of assets to select, by default 5

    Returns:
    --------
    List[str]
        List of selected asset symbols
    """
    # Drop NaN values and sort by returns
    valid_returns = rolling_returns.dropna()
    sorted_returns = valid_returns.sort_values(ascending=False)

    if len(sorted_returns) == 0:
        logging.warning("No valid returns data available. Selecting randomly.")
        # Select random assets if no data at all
        selected = np.random.choice(
            rolling_returns.index, size=min(5, len(rolling_returns)), replace=False
        )
        return list(selected)

    # Always select top 5 assets
    selected = sorted_returns.head(5).index

    return list(selected)


def calculate_weights(selected_assets: List[str], max_weight: float = 0.2) -> pd.Series:
    """
    Calculate portfolio weights ensuring no asset exceeds max_weight.

    Parameters:
    -----------
    selected_assets : List[str]
        List of selected asset symbols
    max_weight : float, optional
        Maximum weight for any asset, by default 0.2 (20%)

    Returns:
    --------
    pd.Series
        Portfolio weights
    """
    n_assets = len(selected_assets)
    if n_assets == 0:
        raise ValueError("No assets selected")

    # If we have 5 or fewer assets, each gets max_weight
    if n_assets <= 5:
        return pd.Series(max_weight, index=selected_assets)

    # Calculate equal weights
    weights = pd.Series(1.0 / n_assets, index=selected_assets)

    # Adjust weights if any exceed max_weight
    while weights.max() > max_weight:
        # Find assets exceeding max_weight
        excess = weights[weights > max_weight]
        remaining = weights[weights <= max_weight]

        # Set excess assets to max_weight
        weights[excess.index] = max_weight

        # If no remaining assets, distribute remaining weight equally among all assets
        if len(remaining) == 0:
            weights = pd.Series(max_weight, index=selected_assets)
            break

        # Redistribute remaining weight equally among other assets
        remaining_weight = 1.0 - (max_weight * len(excess))
        weights[remaining.index] = remaining_weight / len(remaining)

    return weights


def create_returns_based_portfolio(
    returns: pd.DataFrame,
    rebalance_frequency: str = "Q",
    lookback_window: int = 63,
    max_weight: float = 0.2,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Create and manage a returns-based portfolio.

    Parameters:
    -----------
    returns : pd.DataFrame
        Daily returns data
    rebalance_frequency : str, optional
        Rebalancing frequency ('Q' for quarterly), by default "Q"
    lookback_window : int, optional
        Lookback window for returns calculation in days, by default 63 (3 months)
    max_weight : float, optional
        Maximum weight for any asset, by default 0.2 (20%)

    Returns:
    --------
    Tuple[pd.Series, Dict[str, Any]]
        (portfolio_returns, portfolio_info)
    """
    # Initialize portfolio
    portfolio_values = pd.Series(index=returns.index)
    portfolio_values.iloc[0] = 1.0  # Start with $1

    # Get rebalancing dates
    freq_map = {
        "D": "D",
        "W": "W",
        "M": "ME",  # Month End
        "Q": "QE",  # Quarter End
        "A": "YE",  # Year End
    }
    rebalance_dates = pd.date_range(
        start=returns.index.min(),
        end=returns.index.max(),
        freq=freq_map.get(rebalance_frequency, rebalance_frequency),
    )

    # Add start date if needed
    if returns.index[0] not in rebalance_dates:
        rebalance_dates = rebalance_dates.insert(0, returns.index[0])

    # Track portfolio info
    portfolio_info = {
        "rebalance_dates": [],
        "selected_assets": [],
        "weights": [],
        "selected_returns": [],  # Returns of selected assets
        "near_miss_assets": [],  # Next 3 highest returns not selected
        "near_miss_returns": [],  # Returns of near-miss assets
    }

    # Initialize holdings
    current_weights = None
    holdings = None

    # Calculate daily portfolio values
    for i, date in enumerate(returns.index[1:], 1):
        # Check if this is a rebalancing date
        if date in rebalance_dates:
            # Calculate rolling returns up to previous day
            lookback_data = returns.loc[: returns.index[i - 1]]
            rolling_returns = calculate_rolling_returns(lookback_data, lookback_window)
            latest_returns = rolling_returns.iloc[-1]

            # Select top performers
            selected_assets = select_top_performers(latest_returns)
            current_weights = calculate_weights(selected_assets, max_weight)

            # Get returns for selected assets
            selected_returns = latest_returns[selected_assets].to_dict()

            # Get next 3 highest returns not selected
            all_assets = latest_returns.sort_values(ascending=False)
            near_miss_assets = []
            near_miss_returns = {}
            count = 0
            for asset, ret in all_assets.items():
                if asset not in selected_assets:
                    near_miss_assets.append(asset)
                    near_miss_returns[asset] = ret
                    count += 1
                    if count >= 3:
                        break

            # Initialize holdings
            total_value = portfolio_values.iloc[i - 1]
            holdings = current_weights * total_value

            # Log rebalancing
            logging.info(f"Rebalancing on {date.date()}")
            logging.info(f"Selected assets: {selected_assets}")
            logging.info(f"Portfolio weights: {current_weights.to_dict()}")
            logging.info(f"Selected returns: {selected_returns}")
            logging.info(f"Near miss assets: {near_miss_assets}")
            logging.info(f"Near miss returns: {near_miss_returns}")

            # Store portfolio info
            portfolio_info["rebalance_dates"].append(date)
            portfolio_info["selected_assets"].append(selected_assets)
            portfolio_info["weights"].append(current_weights.to_dict())
            portfolio_info["selected_returns"].append(selected_returns)
            portfolio_info["near_miss_assets"].append(near_miss_assets)
            portfolio_info["near_miss_returns"].append(near_miss_returns)

        # Update holdings based on price changes
        if holdings is not None:
            price_changes = 1 + returns.loc[date, holdings.index]
            holdings *= price_changes
            portfolio_values.loc[date] = holdings.sum()
        else:
            # Before first rebalance, use equal weights
            portfolio_values.loc[date] = portfolio_values.iloc[i - 1]

    # Calculate daily returns
    portfolio_returns = portfolio_values.pct_change().dropna()

    return portfolio_returns, portfolio_info


def main(
    returns_file: str,
    output_dir: str,
    rebalance_frequency: str = "Q",
    lookback_window: int = 63,
    max_weight: float = 0.2,
) -> None:
    """
    Main function to run the returns-based portfolio strategy.

    Parameters:
    -----------
    returns_file : str
        Path to the CSV file containing returns data
    output_dir : str
        Directory to save output files
    rebalance_frequency : str, optional
        Rebalancing frequency, by default "Q"
    lookback_window : int, optional
        Lookback window for returns calculation, by default 63
    max_weight : float, optional
        Maximum weight for any asset, by default 0.2
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load returns data
    returns = load_returns(returns_file)
    if returns is None:
        return

    # Create portfolio
    portfolio_returns, portfolio_info = create_returns_based_portfolio(
        returns,
        rebalance_frequency=rebalance_frequency,
        lookback_window=lookback_window,
        max_weight=max_weight,
    )

    # Save portfolio returns
    output_file = os.path.join(output_dir, "returns_algorithm_portfolio.csv")
    portfolio_returns.to_csv(output_file)
    logging.info(f"Saved portfolio returns to {output_file}")

    # Save portfolio info
    info_file = os.path.join(output_dir, "returns_algorithm_info.csv")
    pd.DataFrame(portfolio_info).to_csv(info_file)
    logging.info(f"Saved portfolio info to {info_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run returns-based portfolio selection strategy"
    )
    parser.add_argument(
        "--returns", "-r", required=True, help="Path to the returns CSV file"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Directory to save output files"
    )
    parser.add_argument(
        "--rebalance",
        "-f",
        default="Q",
        help="Rebalancing frequency (D, W, M, Q, A)",
    )
    parser.add_argument(
        "--lookback",
        "-l",
        type=int,
        default=63,
        help="Lookback window in days for returns calculation",
    )
    parser.add_argument(
        "--max-weight",
        "-w",
        type=float,
        default=0.2,
        help="Maximum weight for any asset",
    )

    args = parser.parse_args()

    main(
        args.returns,
        args.output,
        rebalance_frequency=args.rebalance,
        lookback_window=args.lookback,
        max_weight=args.max_weight,
    )
