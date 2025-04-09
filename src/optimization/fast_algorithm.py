"""
Fast Algorithm Portfolio Optimization Module

This module implements the Fast Algorithm for portfolio optimization
as described in the Markowitz Mean-Variance Optimization framework.
"""

import argparse
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Import period configurations


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


def filter_by_period(
    returns: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Filter returns data by date range.

    Parameters:
    -----------
    returns : pd.DataFrame
        Returns data
    start_date : Optional[str], optional
        Start date (YYYY-MM-DD), by default None
    end_date : Optional[str], optional
        End date (YYYY-MM-DD), by default None

    Returns:
    --------
    pd.DataFrame
        Filtered returns data
    """
    filtered = returns.copy()

    if start_date:
        # Convert string date to datetime for proper comparison
        start_dt = pd.to_datetime(start_date)
        filtered = filtered[filtered.index >= start_dt]

    if end_date:
        # Convert string date to datetime for proper comparison
        end_dt = pd.to_datetime(end_date)
        filtered = filtered[filtered.index <= end_dt]

    print(f"Filtered returns: {start_date} to {end_date}, rows: {len(filtered)}")
    return filtered


def prepare_optimization_inputs(
    returns: pd.DataFrame, periods_per_year: int = 252
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Prepare inputs for portfolio optimization.

    Parameters:
    -----------
    returns : pd.DataFrame
        Returns data
    periods_per_year : int, optional
        Number of periods per year for annualization, by default 252 (daily)

    Returns:
    --------
    Tuple[pd.Series, pd.DataFrame]
        (expected_returns, covariance_matrix)
    """
    # Expected returns (annualized)
    expected_returns = returns.mean() * periods_per_year

    # Covariance matrix (annualized)
    covariance_matrix = returns.cov() * periods_per_year

    print(f"Prepared optimization inputs: {len(expected_returns)} assets")
    return expected_returns, covariance_matrix


def fast_algorithm_portfolio(
    returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
    target_return: Optional[float] = None,
    long_only: bool = True,
    max_weight: Optional[float] = None,
    min_weight: Optional[float] = None,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Implements the Fast Algorithm for portfolio optimization.

    Parameters:
    -----------
    returns : pd.Series
        Expected returns for each asset
    cov_matrix : pd.DataFrame
        Covariance matrix of returns
    risk_free_rate : float, optional
        Risk-free rate (annualized), by default 0.0
    target_return : Optional[float], optional
        Target portfolio return (if None, maximize Sharpe ratio), by default None
    long_only : bool, optional
        Whether to enforce long-only constraint, by default True
    max_weight : Optional[float], optional
        Maximum weight for any asset, by default None
    min_weight : Optional[float], optional
        Minimum weight for any asset if included, by default None

    Returns:
    --------
    Tuple[pd.Series, Dict[str, Any]]
        (optimal_weights, portfolio_info)
    """
    n = len(returns)

    # If target_return is None, find the tangency portfolio (maximum Sharpe ratio)
    if target_return is None:
        # Define the negative Sharpe ratio as the objective function (to minimize)
        def objective(weights: np.ndarray) -> float:
            portfolio_return = np.sum(returns * weights)
            portfolio_volatility = np.sqrt(
                np.dot(weights.T, np.dot(cov_matrix, weights))
            )
            sharpe = (portfolio_return - risk_free_rate) / portfolio_volatility
            return -sharpe  # Negative because we're minimizing
    else:
        # Define the objective function (minimize portfolio variance)
        def objective(weights: np.ndarray) -> float:
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            return portfolio_variance

    # Define constraints
    constraints: List[Dict[str, Any]] = []

    # Budget constraint (weights sum to 1)
    constraints.append({"type": "eq", "fun": lambda x: np.sum(x) - 1})

    # Return constraint (if target_return is specified)
    if target_return is not None:
        constraints.append(
            {"type": "eq", "fun": lambda x: np.sum(returns * x) - target_return}
        )

    # Initial guess (equal weights)
    init_guess = np.ones(n) / n

    # Bounds (weight constraints)
    if long_only:
        if max_weight is not None:
            bounds = [(0, max_weight) for _ in range(n)]
        else:
            bounds = [(0, 1) for _ in range(n)]
    else:
        if max_weight is not None:
            bounds = [(-max_weight, max_weight) for _ in range(n)]
        else:
            bounds = [(None, None) for _ in range(n)]

    # Run optimization
    result = minimize(
        objective, init_guess, method="SLSQP", bounds=bounds, constraints=constraints
    )

    # Get optimal weights and round very small weights to zero
    optimal_weights = pd.Series(result["x"], index=returns.index)

    # Apply minimum weight constraint (post-optimization)
    if min_weight is not None or max_weight is not None:
        # Iteratively adjust weights to satisfy both min and max constraints
        # This may take multiple iterations as enforcing one constraint can violate another
        for _ in range(10):  # Limit iterations to avoid infinite loops
            changed = False

            # Apply minimum weight constraint
            if min_weight is not None:
                # Set weights below min_weight to 0
                tiny_weights = (optimal_weights > 0) & (optimal_weights < min_weight)
                if any(tiny_weights):
                    optimal_weights[tiny_weights] = 0
                    changed = True

            # Rescale weights to sum to 1
            if optimal_weights.sum() > 0:
                optimal_weights = optimal_weights / optimal_weights.sum()

            # Apply maximum weight constraint after rescaling
            if max_weight is not None:
                # Check if any weight exceeds max_weight
                excess_weights = optimal_weights > max_weight
                if any(excess_weights):
                    # Set excess weights to max_weight
                    excess_amount = sum(optimal_weights[excess_weights] - max_weight)
                    optimal_weights[excess_weights] = max_weight

                    # Distribute excess amount proportionally to non-maxed weights
                    non_max_weights = ~excess_weights & (optimal_weights > 0)
                    if any(non_max_weights):
                        # Calculate how much we can add to each non-maxed weight
                        available_weights = optimal_weights[non_max_weights]
                        available_capacity = sum(max_weight - available_weights)

                        if available_capacity > 0:
                            # Distribute excess proportionally to available capacity
                            distribution_ratio = min(
                                1, excess_amount / available_capacity
                            )
                            optimal_weights[non_max_weights] += distribution_ratio * (
                                max_weight - available_weights
                            )
                            changed = True

            # If no changes were made in this iteration, we're done
            if not changed:
                break

        # Final rescale to ensure sum to 1
        if optimal_weights.sum() > 0:
            optimal_weights = optimal_weights / optimal_weights.sum()

    # Calculate portfolio metrics
    portfolio_return = np.sum(returns * optimal_weights)
    portfolio_volatility = np.sqrt(
        np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights))
    )
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

    portfolio_info: Dict[str, Any] = {
        "return": portfolio_return,
        "volatility": portfolio_volatility,
        "sharpe_ratio": sharpe_ratio,
        "weights": optimal_weights,
    }

    print(f"Optimization completed successfully. Sharpe ratio: {sharpe_ratio:.4f}")
    return optimal_weights, portfolio_info


def generate_efficient_frontier(
    returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
    points: int = 20,
    long_only: bool = True,
) -> pd.DataFrame:
    """
    Generate the efficient frontier.

    Parameters:
    -----------
    returns : pd.Series
        Expected returns for each asset
    cov_matrix : pd.DataFrame
        Covariance matrix of returns
    risk_free_rate : float, optional
        Risk-free rate, by default 0.0
    points : int, optional
        Number of points on the efficient frontier, by default 20
    long_only : bool, optional
        Whether to enforce long-only constraint, by default True

    Returns:
    --------
    pd.DataFrame
        Efficient frontier points (return, volatility, sharpe)
    """
    # Find minimum variance portfolio
    min_var_weights, min_var_info = fast_algorithm_portfolio(
        returns, cov_matrix, risk_free_rate, target_return=None, long_only=long_only
    )
    min_return = min_var_info["return"]

    # Find maximum return portfolio (investing 100% in the asset with highest return)
    max_return_asset = returns.idxmax()
    max_return = returns[max_return_asset]

    # Generate target returns between min and max
    target_returns = np.linspace(min_return, max_return, points)

    # Calculate efficient frontier portfolios
    efficient_frontier: List[Dict[str, float]] = []

    for target_return in target_returns:
        try:
            _, portfolio_info = fast_algorithm_portfolio(
                returns,
                cov_matrix,
                risk_free_rate,
                target_return=target_return,
                long_only=long_only,
            )

            efficient_frontier.append(
                {
                    "return": portfolio_info["return"],
                    "volatility": portfolio_info["volatility"],
                    "sharpe_ratio": portfolio_info["sharpe_ratio"],
                }
            )
        except Exception as e:
            # Skip if optimization fails
            print(f"Optimization failed for target return {target_return:.4f}: {e}")
            continue

    return pd.DataFrame(efficient_frontier)


def analyze_portfolio_performance(
    weights: pd.Series, historical_returns: pd.DataFrame
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Analyze historical performance of a portfolio with given weights.

    Parameters:
    -----------
    weights : pd.Series
        Portfolio weights
    historical_returns : pd.DataFrame
        Historical asset returns

    Returns:
    --------
    Tuple[pd.Series, Dict[str, Any]]
        (portfolio_returns, performance_metrics)
    """
    # Filter historical returns to include only assets in weights
    common_assets = set(weights.index).intersection(set(historical_returns.columns))
    filtered_weights = weights[list(common_assets)]
    filtered_returns = historical_returns[list(common_assets)]

    # Normalize weights to sum to 1
    filtered_weights = filtered_weights / filtered_weights.sum()

    # Calculate portfolio returns
    portfolio_returns = (filtered_returns * filtered_weights).sum(axis=1)

    # Calculate performance metrics
    cumulative_return = (1 + portfolio_returns).prod() - 1
    annualized_return = (1 + portfolio_returns.mean()) ** 252 - 1
    annualized_volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_volatility

    # Maximum drawdown
    cumulative_returns = (1 + portfolio_returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / running_max) - 1
    max_drawdown = drawdown.min()

    # Value at Risk (95%)
    var_95 = portfolio_returns.quantile(0.05)

    # Conditional VaR (95%)
    cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()

    # Calculate rolling annual returns
    rolling_annual_returns = portfolio_returns.rolling(window=252).apply(
        lambda x: (1 + x).prod() - 1
    )

    performance_metrics: Dict[str, Any] = {
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "rolling_annual_returns": rolling_annual_returns,
    }

    print("Portfolio performance analysis completed")
    return portfolio_returns, performance_metrics


def analyze_portfolio_performance_with_rebalancing(
    returns_data: pd.DataFrame,
    optimization_function: Callable,
    rebalance_frequency: str = "Q",  # 'D' (daily), 'W' (weekly), 'M' (monthly), 'Q' (quarterly), 'A' (annual)
    lookback_window: int = 252,  # Trading days for parameter estimation
    risk_free_rate: float = 0.02,
    max_weight: Optional[float] = 0.25,
    min_weight: Optional[float] = 0.01,
    long_only: bool = True,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Analyze historical performance of a portfolio with periodic rebalancing.

    Parameters:
    -----------
    returns_data : pd.DataFrame
        Historical asset returns
    optimization_function : Callable
        Function that performs the portfolio optimization (e.g., fast_algorithm_portfolio)
    rebalance_frequency : str, optional
        Frequency of rebalancing, by default 'Q' (quarterly)
    lookback_window : int, optional
        Number of trading days to use for parameter estimation, by default 252 (1 year)
    risk_free_rate : float, optional
        Risk-free rate (annualized), by default 0.02
    max_weight : Optional[float], optional
        Maximum weight for any asset, by default 0.25
    min_weight : Optional[float], optional
        Minimum weight for any asset if included, by default 0.01
    long_only : bool, optional
        Whether to enforce long-only constraint, by default True

    Returns:
    --------
    Tuple[pd.Series, Dict[str, Any]]
        (portfolio_returns, performance_metrics)
    """
    # Ensure index is sorted
    returns_data = returns_data.sort_index()

    # Map string frequency codes to pandas date offset objects
    freq_map = {
        "D": "D",
        "W": "W",
        "M": "ME",  # Month End
        "Q": "QE",  # Quarter End
        "A": "YE",  # Year End
    }
    rebal_freq = freq_map.get(rebalance_frequency, rebalance_frequency)

    # Generate rebalancing dates
    rebal_dates = pd.date_range(
        start=returns_data.index.min(), end=returns_data.index.max(), freq=rebal_freq
    )

    # Add first date if not already present
    if returns_data.index[0] not in rebal_dates:
        rebal_dates = rebal_dates.insert(0, returns_data.index[0])

    # Initialize tracking variables
    portfolio_returns = pd.Series(index=returns_data.index)
    current_weights = None
    rebalancing_history = []

    # Iterate through each period between rebalancing dates
    for i in range(len(rebal_dates) - 1):
        current_date = rebal_dates[i]
        next_rebal_date = rebal_dates[i + 1]

        # Get dates in the current period
        period_mask = (returns_data.index >= current_date) & (
            returns_data.index < next_rebal_date
        )
        period_dates = returns_data.index[period_mask]

        if len(period_dates) == 0:
            continue

        # Rebalance at the start of the period
        if current_weights is None or current_date in rebal_dates:
            # Determine lookback window for parameter estimation
            lookback_end = current_date
            lookback_start = returns_data.index[returns_data.index <= lookback_end][
                -min(lookback_window, len(returns_data.index)) :
            ][0]

            # Filter data for the lookback period
            lookback_data = returns_data.loc[lookback_start:lookback_end]

            # Prepare optimization inputs
            expected_returns, cov_matrix = prepare_optimization_inputs(lookback_data)

            # Run optimization
            try:
                optimal_weights, _ = optimization_function(
                    expected_returns,
                    cov_matrix,
                    risk_free_rate=risk_free_rate,
                    long_only=long_only,
                    max_weight=max_weight,
                    min_weight=min_weight,
                )
                current_weights = optimal_weights

                # Record rebalancing event
                rebalancing_history.append(
                    {"date": current_date, "weights": current_weights.to_dict()}
                )

                print(f"Rebalanced portfolio on {current_date.date()}")
            except Exception as e:
                print(f"Optimization failed for date {current_date.date()}: {e}")
                if current_weights is None:
                    # For the first period, if optimization fails, use equal weights
                    current_weights = pd.Series(
                        1 / len(returns_data.columns), index=returns_data.columns
                    )
                # else continue with previous weights

        # Calculate period returns using current weights
        period_returns = returns_data.loc[period_dates]
        for date in period_dates:
            # Filter to common assets between weights and available returns
            common_assets = set(current_weights.index) & set(period_returns.columns)
            filtered_weights = current_weights[list(common_assets)]

            # Normalize weights to sum to 1
            filtered_weights = filtered_weights / filtered_weights.sum()

            # Calculate portfolio return for this day
            if date in period_returns.index:
                day_return = (
                    period_returns.loc[date, list(common_assets)] * filtered_weights
                ).sum()
                portfolio_returns.loc[date] = day_return

    # Clean up any missing values
    portfolio_returns = portfolio_returns.dropna()

    if len(portfolio_returns) == 0:
        raise ValueError(
            "No portfolio returns were calculated. Check your data and rebalancing settings."
        )

    # Calculate performance metrics
    cumulative_return = (1 + portfolio_returns).prod() - 1
    annualized_return = (1 + portfolio_returns.mean()) ** 252 - 1
    annualized_volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

    # Maximum drawdown
    cumulative_returns = (1 + portfolio_returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / running_max) - 1
    max_drawdown = drawdown.min()

    # Value at Risk (95%)
    var_95 = portfolio_returns.quantile(0.05)

    # Conditional VaR (95%)
    cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()

    # Calculate rolling annual returns
    rolling_annual_returns = portfolio_returns.rolling(window=252).apply(
        lambda x: (1 + x).prod() - 1
    )

    performance_metrics = {
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "rolling_annual_returns": rolling_annual_returns,
        "rebalancing_history": rebalancing_history,
    }

    print(
        f"Portfolio performance analysis completed with {len(rebalancing_history)} rebalancing events"
    )
    return portfolio_returns, performance_metrics


def main(
    returns_file: str,
    output_dir: str,
    period: Optional[str] = None,
    risk_free_rate: float = 0.02,
    long_only: bool = True,
    max_weight: float = 0.25,
    min_weight: float = 0.01,
    rebalance: bool = False,
    rebalance_frequency: str = "Q",
    lookback_window: int = 252,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    years: Optional[int] = None,
) -> None:
    """
    Main function to run the Fast Algorithm portfolio optimization.

    Parameters:
    -----------
    returns_file : str
        Path to the CSV file containing returns data
    output_dir : str
        Directory to save the output files
    period : Optional[str], optional
        Period to analyze ('financial_crisis', 'post_crisis', 'recent'), by default None
    risk_free_rate : float, optional
        Risk-free rate (annualized), by default 0.02
    long_only : bool, optional
        Whether to enforce long-only constraint, by default True
    max_weight : float, optional
        Maximum weight for any asset, by default 0.25
    min_weight : float, optional
        Minimum weight for any asset if included, by default 0.01
    rebalance : bool, optional
        Whether to perform periodic rebalancing, by default False
    rebalance_frequency : str, optional
        Frequency of rebalancing ('D', 'W', 'M', 'Q', 'A'), by default 'Q'
    lookback_window : int, optional
        Number of trading days to use for parameter estimation, by default 252
    """
    print("Running Fast Algorithm portfolio optimization")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load returns data
    # Special handling for financial crisis period
    if period == "financial_crisis":
        # Use the specialized financial crisis sector returns that exclude ETFs that didn't exist then
        crisis_file = returns_file.replace(
            "sector_returns.csv", "financial_crisis_sector_returns.csv"
        )
        if os.path.exists(crisis_file):
            print(f"Using specialized financial crisis data from {crisis_file}")
            returns_data = load_returns(crisis_file)
        else:
            # Fall back to regular returns file
            returns_data = load_returns(returns_file)
    else:
        returns_data = load_returns(returns_file)

    if returns_data is None:
        return

    # Define period date ranges
    periods: Dict[str, Tuple[str, str]] = {
        "financial_crisis": ("2008-01-02", "2013-01-02"),
        "post_crisis": ("2012-01-01", "2018-12-31"),
        "recent": ("2019-01-01", "2023-12-31"),
    }

    # Handle custom date range
    if period == "custom":
        # Validate custom date parameters
        if start_date is None:
            print("ERROR: --start-date is required when using --period custom")
            return

        # If end_date is not provided but years is, calculate end_date
        if end_date is None and years is not None:
            try:
                from datetime import datetime, timedelta

                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                end_dt = start_dt + timedelta(days=365 * years)
                end_date = end_dt.strftime("%Y-%m-%d")
                print(
                    f"Calculated end date based on {years} years from {start_date}: {end_date}"
                )
            except ValueError:
                print("ERROR: Invalid start date format. Please use YYYY-MM-DD format.")
                return
        elif end_date is None:
            print(
                "ERROR: Either --end-date or --years must be provided with --period custom"
            )
            return

        # Create specialized data for custom period (similar to financial_crisis handling)
        print(
            f"Preparing specialized data for custom period: {start_date} to {end_date}..."
        )

        # Load raw price data to properly handle ETFs that don't exist during the period
        from src.preprocessing.process_etf_data import prepare_custom_period_data

        custom_data = prepare_custom_period_data(start_date, end_date)

        if custom_data is None or len(custom_data) == 0:
            print(
                f"ERROR: No data available for custom period ({start_date} to {end_date})"
            )
            print("Please choose a different period or check your data source.")
            return

        # Use the custom period data
        returns_data = custom_data
        print(f"Using {len(returns_data.columns)} ETFs for custom period analysis")
    # Filter returns by predefined period if specified
    elif period and period in periods:
        start_date, end_date = periods[period]
        returns_data = filter_by_period(returns_data, start_date, end_date)

        # Check if we have enough data after filtering
        if len(returns_data) == 0:
            print(
                f"ERROR: No data available for period {period} ({start_date} to {end_date})"
            )
            print("Please choose a different period or check your data source.")
            return

    # Prepare optimization inputs
    expected_returns, covariance_matrix = prepare_optimization_inputs(returns_data)

    # Run Fast Algorithm optimization
    optimal_weights, portfolio_info = fast_algorithm_portfolio(
        expected_returns,
        covariance_matrix,
        risk_free_rate,
        long_only=long_only,
        max_weight=max_weight,
        min_weight=min_weight,
    )

    # Run optimization
    if rebalance:
        print(
            f"Using {rebalance_frequency} rebalancing with {lookback_window}-day lookback window"
        )
        (
            portfolio_returns,
            performance_metrics,
        ) = analyze_portfolio_performance_with_rebalancing(
            returns_data,
            fast_algorithm_portfolio,
            rebalance_frequency=rebalance_frequency,
            lookback_window=lookback_window,
            risk_free_rate=risk_free_rate,
            long_only=long_only,
            max_weight=max_weight,
            min_weight=min_weight,
        )

        # Save rebalancing history
        rebalancing_df = pd.DataFrame(performance_metrics["rebalancing_history"])
        rebalancing_file = os.path.join(output_dir, "fa_rebalancing_history.csv")
        rebalancing_df.to_csv(rebalancing_file, index=False)
        print(f"Saved rebalancing history to {rebalancing_file}")
    else:
        # Analyze historical performance
        portfolio_returns, performance_metrics = analyze_portfolio_performance(
            optimal_weights, returns_data
        )

    # Save results
    # Save optimal weights
    weights_file = os.path.join(output_dir, "fa_optimal_weights.csv")
    optimal_weights.to_csv(weights_file)
    print(f"Saved optimal weights to {weights_file}")

    # Save portfolio metrics
    metrics_df = pd.DataFrame(
        {
            "metric": ["return", "volatility", "sharpe_ratio"],
            "value": [
                portfolio_info["return"],
                portfolio_info["volatility"],
                portfolio_info["sharpe_ratio"],
            ],
        }
    )
    metrics_file = os.path.join(output_dir, "fa_portfolio_metrics.csv")
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Saved portfolio metrics to {metrics_file}")

    # Save portfolio returns
    returns_file = os.path.join(output_dir, "fa_portfolio_returns.csv")
    portfolio_returns.to_frame("portfolio_return").to_csv(returns_file)
    print(f"Saved portfolio returns to {returns_file}")

    # Save rolling annual returns for histogram
    rolling_returns_file = os.path.join(output_dir, "fa_rolling_annual_returns.csv")
    performance_metrics["rolling_annual_returns"].to_frame("annual_return").to_csv(
        rolling_returns_file
    )
    print(f"Saved rolling annual returns to {rolling_returns_file}")

    print("\nFast Algorithm optimization complete")
    print(f"Portfolio expected return: {portfolio_info['return']:.4%}")
    print(f"Portfolio expected volatility: {portfolio_info['volatility']:.4%}")
    print(f"Portfolio Sharpe ratio: {portfolio_info['sharpe_ratio']:.4f}")

    # Print allocations (assets with weights > 1%)
    print("\nOptimal Portfolio Allocation:")
    for asset, weight in optimal_weights.items():
        if weight > 0.01:  # Only show allocations > 1%
            print(f"  {asset}: {weight:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fast Algorithm portfolio optimization"
    )
    parser.add_argument(
        "--data", "-d", required=True, help="Path to the returns CSV file"
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
        "--max-weight",
        "-m",
        type=float,
        default=0.25,
        help="Maximum weight for any asset",
    )
    parser.add_argument(
        "--min-weight",
        type=float,
        default=0.01,
        help="Minimum weight for any asset if included",
    )
    parser.add_argument(
        "--allow-short", action="store_true", help="Allow short selling (not long-only)"
    )

    # Add rebalancing arguments
    parser.add_argument(
        "--rebalance-portfolio",
        action="store_true",
        help="Enable periodic rebalancing of the optimized portfolio",
    )
    parser.add_argument(
        "--rebalance-frequency",
        choices=["D", "W", "M", "Q", "A"],
        default="Q",
        help="Frequency for portfolio rebalancing (if enabled)",
    )
    parser.add_argument(
        "--estimation-window",
        type=int,
        default=252,
        help="Lookback window in trading days for parameter estimation",
    )
    args = parser.parse_args()

    main(
        args.data,
        args.output,
        period=args.period,
        risk_free_rate=args.risk_free_rate,
        long_only=not args.allow_short,
        max_weight=args.max_weight,
        min_weight=args.min_weight,
        rebalance=args.rebalance_portfolio,
        rebalance_frequency=args.rebalance_frequency,
        lookback_window=args.estimation_window,
        start_date=args.start_date,
        end_date=args.end_date,
        years=args.years,
    )
