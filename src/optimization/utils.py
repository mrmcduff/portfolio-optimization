import numpy as np
import pandas as pd


def calculate_performance_metrics(portfolio_returns: pd.Series, risk_free_rate: float):
    """
    Calculate portfolio performance metrics given a pandas Series of returns and a risk-free rate.

    Args:
        portfolio_returns (pd.Series): Series of portfolio returns.
        risk_free_rate (float): Risk-free rate for Sharpe ratio calculation.

    Returns:
        dict: Dictionary of performance metrics.
    """
    portfolio_returns = portfolio_returns.dropna()
    if len(portfolio_returns) == 0:
        raise ValueError("No portfolio returns were calculated. Check your data.")

    cumulative_return = (1 + portfolio_returns).prod() - 1
    annualized_return = (1 + portfolio_returns.mean()) ** 252 - 1
    annualized_volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

    cumulative_returns = (1 + portfolio_returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / running_max) - 1
    max_drawdown = drawdown.min()

    var_95 = portfolio_returns.quantile(0.05)
    cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()

    rolling_annual_returns = portfolio_returns.rolling(window=252).apply(
        lambda x: (1 + x).prod() - 1
    )

    return {
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "rolling_annual_returns": rolling_annual_returns,
    }
