"""
One-Factor Fast Algorithm Portfolio Optimization Module

This module implements a portfolio optimization strategy that:
1. Uses the Fast Algorithm selection method to choose stocks
2. Rebalances monthly using 30-day trailing periods
3. Tracks and outputs betas for each stock considered

The core selection method is from fast_algorithm_selection.py, with modifications
to fit into our pipeline and handle monthly rebalancing.
"""

import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd

from src.optimization.fast_algorithm_selection import SingleIndexModel


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
        print(f"Successfully loaded returns data from {file_path}")
        print(f"Shape: {returns.shape}")
        return returns
    except Exception as e:
        print(f"Error loading returns data: {e}")
        return None


def analyze_portfolio_performance_with_rebalancing(
    returns_data: pd.DataFrame,
    market_returns: pd.Series,
    output_dir: str,
    period: str = "recent",
    risk_free_rate: float = 0.02,
    rebalance_frequency: str = "M",
    lookback_window: int = 252,
    years: int = 5,
) -> pd.Series:
    """
    Analyze portfolio performance with periodic rebalancing.

    Parameters:
    -----------
    returns_data : pd.DataFrame
        DataFrame containing returns for all securities
    market_returns : pd.Series
        Series of market returns
    output_dir : str
        Directory to save output files
    period : str, optional
        Time period to analyze, by default "recent"
    risk_free_rate : float, optional
        Annualized risk-free rate, by default 0.02
    rebalance_frequency : str, optional
        Rebalancing frequency, by default "M" (monthly)
    lookback_window : int, optional
        Number of days to look back for parameter estimation, by default 252
    years : int, optional
        Number of years to analyze, by default 5

    Returns:
    --------
    pd.Series
        Series of portfolio returns
    """
    # Map frequency codes to pandas frequency strings
    freq_map = {
        "D": "D",  # Daily
        "W": "W",  # Weekly
        "M": "MS",  # Month Start
        "Q": "QS",  # Quarter Start
        "Y": "YS",  # Year Start
    }

    # Generate rebalancing dates
    start_date = returns_data.index[0]
    end_date = returns_data.index[-1]
    rebal_dates = pd.date_range(
        start=start_date, end=end_date, freq=freq_map[rebalance_frequency]
    )

    # Initialize portfolio tracking
    portfolio_weights = {}
    portfolio_returns = pd.Series(index=returns_data.index, dtype=float)
    portfolio_value = 1.0

    # Track betas for each security
    security_betas = pd.DataFrame(
        index=returns_data.index, columns=returns_data.columns
    )
    security_betas.iloc[0] = 1.0

    # Track period-by-period information
    period_info = []

    # Process each rebalancing period
    for i in range(len(rebal_dates) - 1):
        start_rebal = rebal_dates[i]
        end_rebal = rebal_dates[i + 1]

        # Get lookback data for parameter estimation
        lookback_start = start_rebal - pd.Timedelta(days=lookback_window)
        lookback_data = returns_data.loc[lookback_start:start_rebal]

        # Initialize and run the model
        model = SingleIndexModel(
            returns=lookback_data,
            market_returns=market_returns.loc[lookback_start:start_rebal],
            risk_free_rate=risk_free_rate,
        )
        # Print period dates
        print(
            f"\nPeriod: {start_rebal.strftime('%Y-%m-%d')} to {end_rebal.strftime('%Y-%m-%d')}"
        )

        # Calculate optimal portfolio and get the selected weights
        current_weights = model.calculate_optimal_portfolio()
        print(f"Current weights: {current_weights}")

        # Store weights for this period
        period_weights = current_weights.copy()  # Make a copy to avoid reference issues

        # Calculate portfolio returns for this period
        period_returns = returns_data.loc[start_rebal:end_rebal]
        period_daily_returns = []
        period_cumulative_return = 1.0

        for date in period_returns.index:
            # Store the weights for this date
            portfolio_weights[date] = period_weights

            if date > start_rebal:  # Skip first day as it's already initialized
                # Calculate daily return using the stored weights
                daily_return = sum(
                    weight * period_returns.loc[date, security]
                    for security, weight in period_weights.items()
                    if security in period_returns.columns
                )

                # Update portfolio value and store return
                portfolio_value *= 1 + daily_return
                portfolio_returns.loc[date] = daily_return
                period_daily_returns.append(daily_return)
                period_cumulative_return *= 1 + daily_return

        # Store betas for this period
        for security in returns_data.columns:
            if security in model.parameters:
                security_betas.loc[start_rebal:end_rebal, security] = model.parameters[
                    security
                ]["beta"]

        # Record period information using the actual weights used
        period_info.append(
            {
                "start_date": start_rebal,
                "end_date": end_rebal,
                "selected_securities": list(period_weights.keys()),
                "weights": period_weights,
                "period_return": period_cumulative_return - 1,
                "daily_returns": period_daily_returns,
                "market_return": (1 + market_returns.loc[start_rebal:end_rebal]).prod()
                - 1,
            }
        )

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    # Save portfolio returns
    portfolio_returns.to_csv(
        os.path.join(output_dir, "ofa_portfolio_returns.csv"),
        header=["Return"],
    )

    # Save portfolio weights
    weights_df = pd.DataFrame.from_dict(portfolio_weights, orient="index")
    weights_df.to_csv(os.path.join(output_dir, "ofa_portfolio_weights.csv"))

    # Save security betas
    security_betas.to_csv(os.path.join(output_dir, "ofa_security_betas.csv"))

    # Save period-by-period information
    period_df = pd.DataFrame(period_info)
    period_df.to_csv(os.path.join(output_dir, "ofa_period_analysis.csv"))

    print("Portfolio analysis completed successfully!")

    return portfolio_returns


def main(
    data_path: str,
    market_path: str,
    output_dir: str,
    period: str = "recent",
    risk_free_rate: float = 0.02,
    rebalance_frequency: str = "M",
    lookback_window: int = 252,
    years: int = 5,
) -> None:
    """
    Main function to run the One Factor Fast Algorithm optimization.

    Parameters:
    -----------
    data_path : str
        Path to the returns data file
    market_path : str
        Path to the market returns data file
    output_dir : str
        Directory to save results
    period : str, optional
        Time period to analyze, by default "recent"
    risk_free_rate : float, optional
        Annualized risk-free rate, by default 0.02
    rebalance_frequency : str, optional
        Portfolio rebalancing frequency, by default "M"
    lookback_window : int, optional
        Lookback window for parameter estimation, by default 252
    years : int, optional
        Number of years to analyze, by default 5
    """
    print("Running One-Factor Fast Algorithm portfolio optimization")

    # Load data
    returns_data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    market_data = pd.read_csv(market_path, index_col=0, parse_dates=True)

    print(f"Successfully loaded returns data from {data_path}")
    print(f"Shape: {returns_data.shape}")
    print(f"Successfully loaded returns data from {market_path}")
    print(f"Shape: {market_data.shape}")

    # Run analysis
    portfolio_returns = analyze_portfolio_performance_with_rebalancing(
        returns_data,
        market_data,
        output_dir,
        period=period,
        risk_free_rate=risk_free_rate,
        rebalance_frequency=rebalance_frequency,
        lookback_window=lookback_window,
        years=years,
    )

    print("Portfolio analysis completed successfully!")
    print("\nOne-Factor Fast Algorithm optimization complete")

    # Calculate and print performance metrics
    # Convert daily returns to annualized
    annualized_return = float(
        (1 + portfolio_returns.mean()) ** 252 - 1
    )  # Convert daily to annual
    annualized_vol = float(
        portfolio_returns.std() * np.sqrt(252)
    )  # Annualize volatility

    print(f"Portfolio annualized return: {annualized_return:.4%}")
    print(f"Portfolio annualized volatility: {annualized_vol:.4%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="One-Factor Fast Algorithm portfolio optimization"
    )
    parser.add_argument(
        "--data", "-d", required=True, help="Path to the returns CSV file"
    )
    parser.add_argument(
        "--market", "-m", required=True, help="Path to the market returns CSV file"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Directory to save output files"
    )
    parser.add_argument(
        "--period",
        "-p",
        choices=["financial_crisis", "post_crisis", "recent", "custom"],
        help="Period to analyze (use 'custom' with --start-date and --end-date for custom periods)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for custom period (YYYY-MM-DD format)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date for custom period (YYYY-MM-DD format)",
    )
    parser.add_argument(
        "--years",
        type=int,
        help="Number of years for custom period (starting from start-date)",
    )
    parser.add_argument(
        "--risk-free-rate",
        "-r",
        type=float,
        default=0.02,
        help="Risk-free rate (annualized)",
    )
    parser.add_argument(
        "--rebalance-frequency",
        choices=["D", "W", "M", "Q", "A"],
        default="M",
        help="Frequency for portfolio rebalancing (default: M for monthly)",
    )
    parser.add_argument(
        "--lookback-window",
        type=int,
        default=252,
        help="Lookback window in trading days for parameter estimation (default: 252 days)",
    )
    args = parser.parse_args()

    main(
        args.data,
        args.market,
        args.output,
        period=args.period,
        risk_free_rate=args.risk_free_rate,
        rebalance_frequency=args.rebalance_frequency,
        lookback_window=args.lookback_window,
        years=args.years,
    )
