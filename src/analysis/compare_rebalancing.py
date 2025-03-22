"""
Rebalancing Comparison Module

This script generates and analyzes different rebalancing strategies for a 60/40 portfolio.
"""

import argparse
import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set better plot styles
plt.style.use("ggplot")


def load_returns_data(file_path: str) -> Optional[pd.DataFrame]:
    """Load returns data from CSV file."""
    try:
        returns = pd.read_csv(file_path, index_col=0, parse_dates=True)
        print(f"Successfully loaded returns data from {file_path}")
        print(
            f"Time range: {returns.index.min().date()} to {returns.index.max().date()}"
        )
        print(f"Total days: {len(returns)}")
        return returns
    except Exception as e:
        print(f"Error loading returns data: {e}")
        return None


def create_no_rebalance_portfolio(
    stock_returns: pd.Series, bond_returns: pd.Series, stock_weight: float = 0.6
) -> pd.Series:
    """Create portfolio without rebalancing (weights drift over time)."""
    # Simple weighted returns (no rebalancing)
    returns = stock_returns * stock_weight + bond_returns * (1 - stock_weight)
    return returns


def create_periodic_rebalance_portfolio(
    stock_returns: pd.Series,
    bond_returns: pd.Series,
    stock_weight: float = 0.6,
    freq: str = "M",
) -> pd.Series:
    """Create portfolio with periodic rebalancing."""
    # Create a DataFrame of daily prices starting at $1
    prices = pd.DataFrame(
        {"stock": (1 + stock_returns).cumprod(), "bond": (1 + bond_returns).cumprod()}
    )

    # Initialize portfolio with proper weights
    portfolio_values = pd.Series(index=prices.index)
    portfolio_values.iloc[0] = 1.0  # Start with $1

    # Initialize holdings
    stock_holdings = stock_weight * portfolio_values.iloc[0]
    bond_holdings = (1 - stock_weight) * portfolio_values.iloc[0]

    # Get rebalancing dates
    if freq == "D":
        # Daily rebalancing (every day)
        rebalance_dates = prices.index
    else:
        # Use date_range for other frequencies
        # Change it to use a mapping for frequencies:
        freq_map = {
            "D": "D",
            "W": "W",
            "M": "ME",  # Month End
            "Q": "QE",  # Quarter End
            "A": "YE",  # Year End
        }
        rebalance_dates = pd.date_range(
            start=prices.index.min(),
            end=prices.index.max(),
            freq=freq_map.get(
                freq, freq
            ),  # Use the mapped frequency or the original if not in map
        )

    # Add start date if needed
    if prices.index[0] not in rebalance_dates:
        rebalance_dates = rebalance_dates.insert(0, prices.index[0])

    # Track last rebalance date
    last_rebal_date = prices.index[0]
    rebal_count = 1  # Count first day as a "rebalance"

    # Calculate daily portfolio values
    for i, date in enumerate(prices.index[1:], 1):
        # Update holdings based on price changes
        stock_price_change = (
            prices.loc[date, "stock"] / prices.loc[prices.index[i - 1], "stock"]
        )
        bond_price_change = (
            prices.loc[date, "bond"] / prices.loc[prices.index[i - 1], "bond"]
        )

        stock_holdings *= stock_price_change
        bond_holdings *= bond_price_change

        # Calculate portfolio value
        portfolio_values.loc[date] = stock_holdings + bond_holdings

        # Check if this is a rebalancing date
        if date in rebalance_dates:
            # Rebalance
            total_value = portfolio_values.loc[date]
            stock_holdings = stock_weight * total_value
            bond_holdings = (1 - stock_weight) * total_value
            # last_rebal_date = date
            rebal_count += 1

    # Calculate daily returns
    portfolio_returns = portfolio_values.pct_change().dropna()

    print(f"Created portfolio with {freq} rebalancing ({rebal_count} rebalances)")
    return portfolio_returns


def create_threshold_rebalance_portfolio(
    stock_returns: pd.Series,
    bond_returns: pd.Series,
    stock_weight: float = 0.6,
    threshold: float = 0.05,
) -> pd.Series:
    """Create portfolio with threshold rebalancing."""
    # Create a DataFrame of daily prices starting at $1
    prices = pd.DataFrame(
        {"stock": (1 + stock_returns).cumprod(), "bond": (1 + bond_returns).cumprod()}
    )

    # Initialize portfolio with proper weights
    portfolio_values = pd.Series(index=prices.index)
    portfolio_values.iloc[0] = 1.0  # Start with $1

    # Initialize holdings
    stock_holdings = stock_weight * portfolio_values.iloc[0]
    bond_holdings = (1 - stock_weight) * portfolio_values.iloc[0]

    # Track rebalancing
    rebal_dates = [prices.index[0]]  # First day counts as a "rebalance"

    # Calculate daily portfolio values
    for i, date in enumerate(prices.index[1:], 1):
        # Update holdings based on price changes
        stock_price_change = (
            prices.loc[date, "stock"] / prices.loc[prices.index[i - 1], "stock"]
        )
        bond_price_change = (
            prices.loc[date, "bond"] / prices.loc[prices.index[i - 1], "bond"]
        )

        stock_holdings *= stock_price_change
        bond_holdings *= bond_price_change

        # Calculate portfolio value
        total_value = stock_holdings + bond_holdings
        portfolio_values.loc[date] = total_value

        # Check current allocation vs target
        current_stock_weight = stock_holdings / total_value

        # Check if threshold is breached
        if abs(current_stock_weight - stock_weight) > threshold:
            # Rebalance
            stock_holdings = stock_weight * total_value
            bond_holdings = (1 - stock_weight) * total_value
            rebal_dates.append(date)

    # Calculate daily returns
    portfolio_returns = portfolio_values.pct_change().dropna()

    print(
        f"Created portfolio with {threshold:.1%} threshold rebalancing ({len(rebal_dates)} rebalances)"
    )
    return portfolio_returns


def analyze_portfolios(
    portfolios: Dict[str, pd.Series],
    output_dir: Optional[str] = None,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """Analyze and compare portfolio performance."""
    # Ensure all series are aligned
    returns_df = pd.DataFrame(portfolios)

    # Calculate cumulative returns
    cumulative_returns = (1 + returns_df).cumprod()

    # Calculate performance metrics
    metrics = {}
    for name, returns in portfolios.items():
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        ann_return = (1 + returns.mean()) ** 252 - 1
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = (ann_return - risk_free_rate) / ann_vol

        # Drawdown analysis
        wealth_index = (1 + returns).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        max_drawdown = drawdowns.min()

        # Value at Risk
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean()

        metrics[name] = {
            "Total Return": total_return,
            "Annualized Return": ann_return,
            "Annualized Volatility": ann_vol,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_drawdown,
            "VaR (95%)": var_95,
            "CVaR (95%)": cvar_95,
        }

    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics).T

    # Create plots if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Plot cumulative returns
        plt.figure(figsize=(12, 8))
        for col in cumulative_returns.columns:
            plt.plot(cumulative_returns.index, cumulative_returns[col], label=col)

        plt.title("Cumulative Returns by Rebalancing Strategy")
        plt.xlabel("Date")
        plt.ylabel("Growth of $1 (log scale)")
        plt.legend()
        plt.yscale("log")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "cumulative_returns.png"), dpi=300)

        # Plot drawdowns
        plt.figure(figsize=(12, 8))
        for col in returns_df.columns:
            wealth_index = (1 + returns_df[col]).cumprod()
            previous_peaks = wealth_index.cummax()
            drawdowns = (wealth_index - previous_peaks) / previous_peaks
            plt.plot(drawdowns.index, drawdowns, label=col)

        plt.title("Drawdowns by Rebalancing Strategy")
        plt.xlabel("Date")
        plt.ylabel("Drawdown")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "drawdowns.png"), dpi=300)

        # Save metrics to CSV
        metrics_df.to_csv(os.path.join(output_dir, "rebalancing_metrics.csv"))

        # Rebalancing comparison report
        report = "# Rebalancing Strategy Comparison\n\n"
        report += f"Analysis period: {returns_df.index.min().date()} to {returns_df.index.max().date()}\n\n"
        report += "## Performance Metrics\n\n"

        # Format metrics as percentages where appropriate
        formatted_metrics = metrics_df.copy()
        for col in [
            "Total Return",
            "Annualized Return",
            "Annualized Volatility",
            "Max Drawdown",
            "VaR (95%)",
            "CVaR (95%)",
        ]:
            formatted_metrics[col] = formatted_metrics[col].map(lambda x: f"{x:.2%}")

        # Add metrics table
        report += formatted_metrics.to_markdown() + "\n\n"

        # Add explanations
        report += "## Understanding Rebalancing Strategies\n\n"
        report += "### No Rebalancing\n"
        report += "Portfolio weights drift with market movements, which typically increases equity exposure over time in bull markets.\n\n"

        report += "### Monthly Rebalancing\n"
        report += "Portfolio is rebalanced to target weights at the end of each calendar month, maintaining consistent risk exposure.\n\n"

        report += "### Quarterly Rebalancing\n"
        report += "Portfolio is rebalanced every three months, balancing transaction costs with risk control.\n\n"

        report += "### Threshold Rebalancing\n"
        report += "Portfolio is rebalanced only when allocations drift more than 5% from targets, potentially reducing unnecessary transactions.\n\n"

        report += "## Analysis Insights\n\n"

        # Add some basic insights based on results
        best_return = metrics_df["Annualized Return"].idxmax()
        best_sharpe = metrics_df["Sharpe Ratio"].idxmax()
        lowest_drawdown = metrics_df["Max Drawdown"].idxmax()  # Less negative = better

        report += f"- **Best Return**: {best_return} with {metrics_df.loc[best_return, 'Annualized Return']:.2%} annualized\n"
        report += f"- **Best Risk-Adjusted Return**: {best_sharpe} with Sharpe ratio of {metrics_df.loc[best_sharpe, 'Sharpe Ratio']:.2f}\n"
        report += f"- **Lowest Drawdown**: {lowest_drawdown} with maximum drawdown of {metrics_df.loc[lowest_drawdown, 'Max Drawdown']:.2%}\n\n"

        report += "## Rebalancing Implications\n\n"
        report += "During trending markets, less frequent rebalancing or no rebalancing may capture more upside from the better-performing asset class. "
        report += "However, this comes at the cost of increased risk from higher concentration.\n\n"
        report += "More frequent rebalancing provides better risk control and more consistent exposure to the investor's target allocation. "
        report += "This approach may underperform during strong trending markets but offers better protection during periods of mean reversion.\n\n"
        report += "Threshold-based rebalancing attempts to balance these concerns by only trading when allocations drift significantly, "
        report += "potentially reducing transaction costs while still providing reasonable risk control.\n"

        # Save report
        with open(os.path.join(output_dir, "rebalancing_comparison.md"), "w") as f:
            f.write(report)

    return metrics_df


def main(
    stock_symbol: str,
    bond_symbol: str,
    returns_file: str,
    output_dir: str,
    stock_weight: float = 0.6,
    risk_free_rate: float = 0.02,
    threshold: float = 0.05,
) -> None:
    """Generate and compare different rebalancing strategies."""
    # Configure pandas to show all rows and columns
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)  # Wider display to prevent line wrapping

    print(
        f"Comparing rebalancing strategies for {stock_weight:.0%}/{(1 - stock_weight):.0%} {stock_symbol}/{bond_symbol} portfolio"
    )

    # Load returns data
    returns_data = load_returns_data(returns_file)
    if returns_data is None:
        return

    # Ensure we have the required stocks
    if (
        stock_symbol not in returns_data.columns
        or bond_symbol not in returns_data.columns
    ):
        print(
            f"Error: Required symbols ({stock_symbol} and/or {bond_symbol}) not found in returns data"
        )
        return

    # Get returns series
    stock_returns = returns_data[stock_symbol]
    bond_returns = returns_data[bond_symbol]

    # Create different portfolios
    portfolios = {}

    # Add S&P 500 for reference
    portfolios["S&P 500"] = stock_returns

    # No rebalancing
    portfolios["60/40 No Rebalancing"] = create_no_rebalance_portfolio(
        stock_returns, bond_returns, stock_weight
    )

    # Monthly rebalancing
    portfolios["60/40 Monthly Rebalancing"] = create_periodic_rebalance_portfolio(
        stock_returns, bond_returns, stock_weight, "M"
    )

    # Quarterly rebalancing
    portfolios["60/40 Quarterly Rebalancing"] = create_periodic_rebalance_portfolio(
        stock_returns, bond_returns, stock_weight, "Q"
    )

    # Threshold rebalancing
    portfolios["60/40 Threshold Rebalancing"] = create_threshold_rebalance_portfolio(
        stock_returns, bond_returns, stock_weight, threshold
    )

    # Analyze portfolios
    metrics = analyze_portfolios(portfolios, output_dir, risk_free_rate)

    # Display results
    print("\nPerformance Metrics:")
    print(metrics)

    if output_dir:
        print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare different portfolio rebalancing strategies"
    )
    parser.add_argument(
        "--returns", "-r", required=True, help="Path to the returns CSV file"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Directory to save results"
    )
    parser.add_argument("--stock", "-s", default="SPY", help="Stock ETF symbol")
    parser.add_argument("--bond", "-b", default="BND", help="Bond ETF symbol")
    parser.add_argument(
        "--weight", "-w", type=float, default=0.6, help="Target stock weight"
    )
    parser.add_argument(
        "--risk-free-rate", type=float, default=0.02, help="Risk-free rate (annualized)"
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=0.05, help="Rebalancing threshold"
    )

    args = parser.parse_args()

    main(
        args.stock,
        args.bond,
        args.returns,
        args.output,
        args.weight,
        args.risk_free_rate,
        args.threshold,
    )
