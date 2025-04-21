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

from src.config.periods import get_period_range
from src.optimization.fast_algorithm_selection import SingleIndexModel

# Ensure Excel writer engine is available
try:
    import openpyxl  # noqa: F401
except ImportError:
    pass


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
    start_date: str = None,
    end_date: str = None,
    verbose: bool = False,
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
    start_date : str, optional
        Start date for custom period (YYYY-MM-DD format)
    end_date : str, optional
        End date for custom period (YYYY-MM-DD format)

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

    # Generate rebalancing dates starting from the portfolio start date
    available_dates = returns_data.index
    # Determine the true portfolio start date (after trailing data is included)
    if start_date:
        portfolio_start_date = pd.to_datetime(start_date)
        # Align to first available trading day >= start_date
        if (available_dates >= portfolio_start_date).any():
            portfolio_start_date = available_dates[
                available_dates >= portfolio_start_date
            ][0]
        else:
            portfolio_start_date = available_dates[0]
    else:
        portfolio_start_date = available_dates[0]
    rebal_dates = pd.date_range(
        start=portfolio_start_date,
        end=available_dates[-1],
        freq=freq_map[rebalance_frequency],
    )
    # Align each rebalancing date to the closest available trading day ON OR AFTER the intended date
    rebal_dates = [
        available_dates[available_dates >= d][0]
        if (available_dates >= d).any()
        else available_dates[-1]
        for d in rebal_dates
    ]
    # Ensure the period start date is included as the first rebalance
    if rebal_dates[0] != portfolio_start_date:
        rebal_dates = [portfolio_start_date] + rebal_dates
    else:
        rebal_dates = list(rebal_dates)

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
        # Align lookback_end to available dates
        lookback_end = start_rebal
        if lookback_end not in available_dates:
            lookback_end = available_dates[available_dates <= lookback_end][-1]
        # Find the index of lookback_end in returns_data
        end_idx = available_dates.get_loc(lookback_end)
        lookback_len = min(lookback_window, end_idx + 1)
        lookback_start = available_dates[end_idx - lookback_len + 1]
        lookback_data = returns_data.loc[lookback_start:lookback_end]
        # Require a minimum lookback window of 30 days
        min_lookback_window = 30
        # Log if the lookback window is shorter than intended
        if lookback_data.shape[0] < lookback_window:
            print(
                f"[REBALANCE WARNING] {start_rebal.date()}: lookback window shorter than intended ({lookback_data.shape[0]} < {lookback_window} days)"
            )
        if lookback_data.shape[0] < min_lookback_window:
            print(
                f"[REBALANCE SKIP] {start_rebal.date()}: lookback window too short ({lookback_data.shape[0]} < {min_lookback_window} days), skipping rebalance."
            )
            continue
        # Exclude securities with incomplete data in the lookback window
        valid_securities = lookback_data.columns[lookback_data.notna().all(axis=0)]
        lookback_data_valid = lookback_data[valid_securities]

        # Align market data to lookback window
        lookback_market = market_returns.loc[lookback_data.index]
        # Check for sufficient and non-constant market data
        if (
            lookback_market.isna().all()
            or lookback_market.var() == 0
            or lookback_market.count() < 2
        ):
            print(
                f"[REBALANCE SKIP] {start_rebal.date()}: insufficient or constant market data for lookback window (count={lookback_market.count()}, variance={lookback_market.var()})"
            )
            continue
        dropped = set(lookback_data.columns) - set(valid_securities)
        lookback_window_str = f"{lookback_start.date()} to {lookback_end.date()} (length: {lookback_data.shape[0]} days)"
        if lookback_data_valid.shape[1] == 0:
            print(
                f"[REBALANCE] {start_rebal.date()}: lookback {lookback_window_str} -- NO DATA to make allocation (skipped)"
            )
            continue
        else:
            if dropped:
                print(
                    f"[REBALANCE] {start_rebal.date()}: lookback {lookback_window_str} -- allocation made on {lookback_data_valid.shape[1]} securities (dropped: {dropped})"
                )
            else:
                print(
                    f"[REBALANCE] {start_rebal.date()}: lookback {lookback_window_str} -- allocation made on {lookback_data_valid.shape[1]} securities (none dropped)"
                )
        # Logging for rebalance efforts
        if lookback_data_valid.shape[1] == 0:
            print(
                f"[ERROR] No securities with complete data for lookback window ending {lookback_end.date()}. Skipping rebalance."
            )
            continue
        # Initialize and run the model
        model = SingleIndexModel(
            returns=lookback_data_valid,
            market_returns=market_returns.loc[lookback_start:lookback_end],
            risk_free_rate=risk_free_rate,
            output_dir=output_dir,
            verbose=verbose,
        )
        # Calculate optimal portfolio and get the selected weights
        try:
            current_weights = model.calculate_optimal_portfolio()
        except Exception as e:
            print(f"Optimization failed for period starting {start_rebal.date()}: {e}")
            raise RuntimeError(
                f"Optimization failed for period starting {start_rebal.date()}: {e}"
            )

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

    # Filter outputs for reporting if start_date/end_date are specified
    filtered_index = portfolio_returns.index
    if start_date:
        filtered_index = filtered_index[filtered_index >= pd.to_datetime(start_date)]
    if end_date:
        filtered_index = filtered_index[filtered_index <= pd.to_datetime(end_date)]

    # Save portfolio returns
    portfolio_returns.loc[filtered_index].to_csv(
        os.path.join(output_dir, "ofa_portfolio_returns.csv"),
        header=["Return"],
    )
    # Also save as XLSX
    portfolio_returns.loc[filtered_index].to_frame("Return").to_excel(
        os.path.join(output_dir, "ofa_portfolio_returns.xlsx")
    )

    # Save portfolio weights
    weights_df = pd.DataFrame.from_dict(portfolio_weights, orient="index")
    weights_idx = weights_df.index.intersection(filtered_index)
    weights_df.loc[weights_idx].to_csv(
        os.path.join(output_dir, "ofa_portfolio_weights.csv")
    )
    # Also save as XLSX
    weights_df.loc[weights_idx].to_excel(
        os.path.join(output_dir, "ofa_portfolio_weights.xlsx")
    )

    # Save security betas
    betas_idx = security_betas.index.intersection(filtered_index)
    security_betas.loc[betas_idx].to_csv(
        os.path.join(output_dir, "ofa_security_betas.csv")
    )
    # Also save as XLSX
    security_betas.loc[betas_idx].to_excel(
        os.path.join(output_dir, "ofa_security_betas.xlsx")
    )

    # Save period-by-period information
    period_df = pd.DataFrame(period_info)
    period_df = period_df[period_df["start_date"].isin(filtered_index)]
    period_df.to_csv(os.path.join(output_dir, "ofa_period_analysis.csv"))
    # Also save as XLSX
    period_df.to_excel(os.path.join(output_dir, "ofa_period_analysis.xlsx"))

    # Save rebalancing log as one_factor_fast_rebalancing.xlsx
    rebal_log_cols = [
        "start_date",
        "end_date",
        "selected_securities",
        "weights",
        "period_return",
        "market_return",
    ]
    rebal_log_df = pd.DataFrame(
        [
            {k: v for k, v in period.items() if k in rebal_log_cols}
            for period in period_info
        ]
    )
    rebal_log_df = rebal_log_df[rebal_log_df["start_date"].isin(filtered_index)]
    rebal_log_df.to_excel(
        os.path.join(output_dir, "one_factor_fast_rebalancing.xlsx"), index=False
    )

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
    start_date: str = None,
    end_date: str = None,
    verbose: bool = False,
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
    start_date : str, optional
        Start date for custom period (YYYY-MM-DD format)
    end_date : str, optional
        End date for custom period (YYYY-MM-DD format)
    verbose : bool, optional
        Enable verbose logging, by default False
    """
    print("Running One-Factor Fast Algorithm portfolio optimization")

    # Load data
    returns_data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    market_data = pd.read_csv(market_path, index_col=0, parse_dates=True)

    print(f"Successfully loaded returns data from {data_path}")
    print(f"Shape: {returns_data.shape}")
    print(f"Successfully loaded returns data from {market_path}")
    print(f"Shape: {market_data.shape}")

    # Use period configuration from src.config.periods
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
    # Filter by custom or named period if specified
    if period in ("custom", "recent", "financial_crisis", "post_crisis"):
        # --- Ensure enough trailing data is included for lookback window ---
        if start_date:
            start_dt = pd.to_datetime(start_date)
            if lookback_window > 1:
                earliest_needed = start_dt - pd.Timedelta(
                    days=int(lookback_window * 1.5)
                )
            else:
                earliest_needed = start_dt
            returns_data = returns_data[returns_data.index >= earliest_needed]
            market_data = market_data[market_data.index >= earliest_needed]
            print(
                f"For period '{period}', including trailing data from {earliest_needed.strftime('%Y-%m-%d')} to {end_date}"
            )
        # --- END ---
        # DO NOT filter returns_data/market_data for start_date or end_date here; do this only for outputs after optimization

    # Log the date boundaries used and the actual dates in the filtered data
    print(f"Start date argument: {start_date}")
    print(f"End date argument: {end_date}")
    print(
        f"First returns date used: {returns_data.index[0].strftime('%Y-%m-%d') if not returns_data.empty else 'N/A'}"
    )
    print(
        f"Last returns date used: {returns_data.index[-1].strftime('%Y-%m-%d') if not returns_data.empty else 'N/A'}"
    )

    # Run analysis
    portfolio_returns = analyze_portfolio_performance_with_rebalancing(
        returns_data,
        market_data.squeeze(),
        output_dir,
        period=period,
        risk_free_rate=risk_free_rate,
        rebalance_frequency=rebalance_frequency,
        lookback_window=lookback_window,
        years=years,
        start_date=start_date,
        end_date=end_date,
    )

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
        description="One-Factor Fast Algorithm Portfolio Optimization"
    )
    parser.add_argument("--data", required=True, help="Path to returns data CSV")
    parser.add_argument("--market", required=True, help="Path to market returns CSV")
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument(
        "--period", default="recent", help="Period type: recent or custom"
    )
    parser.add_argument(
        "--risk-free-rate", type=float, default=0.02, help="Annualized risk-free rate"
    )
    parser.add_argument(
        "--rebalance-frequency", default="M", help="Rebalancing frequency (D/W/M/Q/Y)"
    )
    parser.add_argument(
        "--lookback-window", type=int, default=252, help="Lookback window in days"
    )
    parser.add_argument(
        "--years", type=int, default=5, help="Number of years to analyze"
    )
    parser.add_argument(
        "--start-date", type=str, default=None, help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", type=str, default=None, help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging to console"
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
        start_date=getattr(args, "start_date", None),
        end_date=getattr(args, "end_date", None),
        verbose=getattr(args, "verbose", False),
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
        start_date=getattr(args, "start_date", None),
        end_date=getattr(args, "end_date", None),
        verbose=getattr(args, "verbose", False),
    )
