"""
Fast Algorithm Portfolio Optimization Module

This module implements the Fast Algorithm for portfolio optimization
as described in the Markowitz Mean-Variance Optimization framework.
"""

import argparse
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize


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
        filtered = filtered[filtered.index >= start_date]

    if end_date:
        filtered = filtered[filtered.index <= end_date]

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
    if min_weight is not None:
        # Set weights below min_weight to 0
        tiny_weights = optimal_weights < min_weight
        optimal_weights[tiny_weights] = 0

        # Rescale remaining weights to sum to 1
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
        except:
            # Skip if optimization fails
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

    print(f"Portfolio performance analysis completed")
    return portfolio_returns, performance_metrics


def main(
    returns_file: str,
    output_dir: str,
    period: Optional[str] = None,
    risk_free_rate: float = 0.02,
    long_only: bool = True,
    max_weight: float = 0.25,
    min_weight: float = 0.01,
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
    """
    print(f"Running Fast Algorithm portfolio optimization")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load returns data
    returns_data = load_returns(returns_file)
    if returns_data is None:
        return

    # Define period date ranges
    periods: Dict[str, Tuple[str, str]] = {
        "financial_crisis": ("2007-01-01", "2011-12-31"),
        "post_crisis": ("2012-01-01", "2018-12-31"),
        "recent": ("2019-01-01", "2023-12-31"),
    }

    # Filter returns by period if specified
    if period and period in periods:
        start_date, end_date = periods[period]
        returns_data = filter_by_period(returns_data, start_date, end_date)

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

    print(f"\nFast Algorithm optimization complete")
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
        choices=["financial_crisis", "post_crisis", "recent"],
        help="Period to analyze",
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

    args = parser.parse_args()

    main(
        args.data,
        args.output,
        period=args.period,
        risk_free_rate=args.risk_free_rate,
        long_only=not args.allow_short,
        max_weight=args.max_weight,
        min_weight=args.min_weight,
    )
