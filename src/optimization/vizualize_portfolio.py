"""
Portfolio Visualization Module

This module provides visualization tools for portfolio optimization results,
including efficient frontier plots, performance comparisons, and return histograms.
"""

import argparse
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_data(weights_file: str, returns_file: str) -> Tuple[Optional[pd.Series], Optional[pd.DataFrame]]:
    """
    Load portfolio weights and historical returns data.

    Parameters:
    -----------
    weights_file : str
        Path to the CSV file containing portfolio weights
    returns_file : str
        Path to the CSV file containing historical returns

    Returns:
    --------
    Tuple[Optional[pd.Series], Optional[pd.DataFrame]]
        (weights, returns) or (None, None) if an error occurs
    """
    try:
        weights = pd.read_csv(weights_file, index_col=0, header=None, squeeze=True)
        returns = pd.read_csv(returns_file, index_col=0, parse_dates=True)

        print(f"Loaded weights: {len(weights)} assets")
        print(f"Loaded returns: {returns.shape}")
        return weights, returns
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


def plot_efficient_frontier(returns: pd.Series,
                          cov_matrix: pd.DataFrame,
                          risk_free_rate: float = 0.0,
                          weights: Optional[pd.Series] = None,
                          asset_names: Optional[List[str]] = None,
                          long_only: bool = True,
                          output_file: Optional[str] = None) -> plt.Figure:
    """
    Plot the efficient frontier with assets.

    Parameters:
    -----------
    returns : pd.Series
        Expected returns for each asset
    cov_matrix : pd.DataFrame
        Covariance matrix of returns
    risk_free_rate : float, optional
        Risk-free rate, by default 0.0
    weights : Optional[pd.Series], optional
        Optimal portfolio weights, by default None
    asset_names : Optional[List[str]], optional
        Names of assets to highlight, by default None
    long_only : bool, optional
        Whether to enforce long-only constraint, by default True
    output_file : Optional[str], optional
        Path to save the plot, by default None

    Returns:
    --------
    plt.Figure
        Matplotlib figure of the efficient frontier
    """
    # Generate points on the efficient frontier
    from scipy.optimize import minimize

    def portfolio_volatility(weights, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    def portfolio_return(weights, returns):
        return np.sum(returns * weights)

    def optimize_portfolio(target_return, returns, cov_matrix, long_only=True):
        n = len(returns)
        # Objective: minimize portfolio volatility
        args = (cov_matrix,)
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
            {'type': 'eq', 'fun': lambda x: portfolio_return(x, returns) - target_return}  # target return
        ]
        bounds = [(0, 1) if long_only else (None, None) for _ in range(n)]

        result = minimize(portfolio_volatility, np.ones(n) / n, args=args,
                         method='SLSQP', bounds=bounds, constraints=constraints)

        return result['x'], result['fun'], portfolio_return(result['x'], returns)

    # Find the minimum variance portfolio
    def min_variance_portfolio(returns, cov_matrix, long_only=True):
        n = len(returns)
        args = (cov_matrix,)
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = [(0, 1) if long_only else (None, None) for _ in range(n)]

        result = minimize(portfolio_volatility, np.ones(n) / n, args=args,
                         method='SLSQP', bounds=bounds, constraints=constraints)

        return result['x'], result['fun'], portfolio_return(result['x'], returns)

    # Generate efficient frontier points
    min_weights, min_vol, min_ret = min_variance_portfolio(returns, cov_matrix, long_only)

    # Get the maximum return asset
    max_ret_idx = returns.idxmax()
    max_ret = returns[max_ret_idx]

    # Generate target returns
    target_returns = np.linspace(min_ret, max_ret, 50)
    efficient_frontier = []

    for target_ret in target_returns:
        try:
            weights, vol, ret = optimize_portfolio(target_ret, returns, cov_matrix, long_only)
            efficient_frontier.append({'return': ret, 'volatility': vol})
        except:
            continue

    ef_df = pd.DataFrame(efficient_frontier)

    # Calculate individual asset risk/return
    asset_returns = returns
    asset_volatility = np.sqrt(np.diag(cov_matrix))

    # Create plot
    plt.figure(figsize=(12, 8))

    # Plot efficient frontier
    plt.plot(ef_df['volatility'], ef_df['return'], 'b-', linewidth=2, label='Efficient Frontier')

    # Plot individual assets
    plt.scatter(asset_volatility, asset_returns, s=50, c='black', alpha=0.7)

    # Add asset labels if provided
    if asset_names is not None:
        for i, name in enumerate(asset_names):
            plt.annotate(name, (asset_volatility[i], asset_returns[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)

    # Plot optimal portfolio if weights provided
    if weights is not None:
        portfolio_vol = portfolio_volatility(weights.values, cov_matrix)
        portfolio_ret = portfolio_return(weights.values, returns)
        plt.scatter(portfolio_vol, portfolio_ret, s=200, c='red', marker='*',
                  label='Optimal Portfolio')

    # Plot capital market line if risk-free rate provided
    if weights is not None and risk_free_rate is not None:
        # Compute Sharpe ratio
        sharpe = (portfolio_ret - risk_free_rate) / portfolio_vol

        # Plot capital market line
        x_line = np.linspace(0, max(asset_volatility) * 1.2, 100)
        y_line = risk_free_rate + sharpe * x_line
        plt.plot(x_line, y_line, 'r--', label='Capital Market Line')
        plt.scatter(0, risk_free_rate, c='black', marker='o', label='Risk-Free Rate')

    # Set labels and title
    plt.xlabel('Annualized Volatility')
    plt.ylabel('Annualized Expected Return')
    plt.title('Efficient Frontier and Capital Market Line')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Save the plot if output_file provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved efficient frontier plot to {output_file}")

    return plt.gcf()


def plot_portfolio_weights(weights: pd.Series, output_file: Optional[str] = None) -> plt.Figure:
    """
    Plot portfolio allocation weights.

    Parameters:
    -----------
    weights : pd.Series
        Portfolio weights
    output_file : Optional[str], optional
        Path to save the plot, by default None

    Returns:
    --------
    plt.Figure
        Matplotlib figure of the portfolio weights
    """
    # Filter out small weights
    significant_weights = weights[weights > 0.01]
    significant_weights = significant_weights.sort_values(ascending=True)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(significant_weights.index, significant_weights.values * 100)

    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                f'{width:.1f}%', ha='left', va='center')

    plt.xlabel('Allocation (%)')
    plt.title('Portfolio Allocation')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    # Save the plot if output_file provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved portfolio weights plot to {output_file}")

    return plt.gcf()


def plot_rolling_returns_histogram(returns: pd.Series,
                                  window: int = 252,
                                  output_file: Optional[str] = None) -> plt.Figure:
    """
    Plot histogram of rolling annual returns.

    Parameters:
    -----------
    returns : pd.Series
        Daily portfolio returns
    window : int, optional
        Rolling window size (days), by default 252
    output_file : Optional[str], optional
        Path to save the plot, by default None

    Returns:
    --------
    plt.Figure
        Matplotlib figure of the rolling returns histogram
    """
    # Calculate rolling returns
    rolling_returns = returns.rolling(window=window).apply(lambda x: (1 + x).prod() - 1)

    plt.figure(figsize=(12, 6))

    # Plot histogram
    sns.histplot(rolling_returns * 100, kde=True, bins=30)

    # Add statistics
    mean = rolling_returns.mean() * 100
    median = rolling_returns.median() * 100
    min_ret = rolling_returns.min() * 100
    max_ret = rolling_returns.max() * 100

    plt.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.2f}%')
    plt.axvline(median, color='green', linestyle='--', label=f'Median: {median:.2f}%')

    plt.title(f'Distribution of {window/252:.1f}-Year Rolling Returns')
    plt.xlabel('Return (%)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add text box with statistics
    stats_text = (f"Mean: {mean:.2f}%\n"
                 f"Median: {median:.2f}%\n"
                 f"Min: {min_ret:.2f}%\n"
                 f"Max: {max_ret:.2f}%")

    plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
                verticalalignment='top')

    plt.tight_layout()

    # Save the plot if output_file provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved rolling returns histogram to {output_file}")

    return plt.gcf()


def plot_benchmark_comparison(portfolio_returns: pd.Series,
                               benchmark_returns: pd.DataFrame,
                               output_file: Optional[str] = None) -> plt.Figure:
    """
    Plot cumulative performance comparison with benchmarks.

    Parameters:
    -----------
    portfolio_returns : pd.Series
        Portfolio returns
    benchmark_returns : pd.DataFrame
        Benchmark returns
    output_file : Optional[str], optional
        Path to save the plot, by default None

    Returns:
    --------
    plt.Figure
        Matplotlib figure of the performance comparison
    """
    # Calculate cumulative returns
    data = pd.concat([portfolio_returns, benchmark_returns], axis=1)
    cumulative_returns = (1 + data).cumprod()

    plt.figure(figsize=(12, 8))

    # Plot cumulative returns
    for col in cumulative_returns.columns:
        plt.plot(cumulative_returns.index, cumulative_returns[col], label=col)

    plt.ylabel('Cumulative Return (1 = Initial Investment)')
    plt.title('Investment Growth Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add annotations for final values
    for col in cumulative_returns.columns:
        final_value = cumulative_returns[col].iloc[-1]
        plt.annotate(f'{final_value:.2f}x',
                   xy=(cumulative_returns.index[-1], final_value),
                   xytext=(5, 0), textcoords='offset points')

    plt.tight_layout()

    # Save the plot if output_file provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved benchmark comparison plot to {output_file}")

    return plt.gcf()


def plot_risk_metrics(performance_metrics: Dict[str, Any],
                     benchmark_metrics: Optional[Dict[str, Dict[str, Any]]] = None,
                     output_file: Optional[str] = None) -> plt.Figure:
    """
    Plot risk and return metrics comparison.

    Parameters:
    -----------
    performance_metrics : Dict[str, Any]
        Portfolio performance metrics
    benchmark_metrics : Optional[Dict[str, Dict[str, Any]]], optional
        Benchmark performance metrics, by default None
    output_file : Optional[str], optional
        Path to save the plot, by default None

    Returns:
    --------
    plt.Figure
        Matplotlib figure of the risk metrics comparison
    """
    # Create DataFrame for plotting
    metrics = ['annualized_return', 'annualized_volatility', 'sharpe_ratio', 'max_drawdown', 'var_95']
    labels = ['Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown', 'VaR (95%)']

    portfolio_data = [performance_metrics[m] for m in metrics]
    data = {'Metric': labels, 'Portfolio': portfolio_data}

    if benchmark_metrics:
        for name, metrics_dict in benchmark_metrics.items():
            data[name] = [metrics_dict[m] for m in metrics]

    df = pd.DataFrame(data).set_index('Metric')

    # Create subplots for different metrics
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Plot return
    df.loc['Return'].plot(kind='bar', ax=axes[0], color='skyblue')
    axes[0].set_title('Annualized Return')
    axes[0].set_ylabel('Return (%)')
    axes[0].grid(axis='y', alpha=0.3)

    # Plot volatility
    df.loc['Volatility'].plot(kind='bar', ax=axes[1], color='orange')
    axes[1].set_title('Annualized Volatility')
    axes[1].set_ylabel('Volatility (%)')
    axes[1].grid(axis='y', alpha=0.3)

    # Plot Sharpe ratio
    df.loc['Sharpe Ratio'].plot(kind='bar', ax=axes[2], color='green')
    axes[2].set_title('Sharpe Ratio')
    axes[2].grid(axis='y', alpha=0.3)

    # Plot max drawdown
    df.loc['Max Drawdown'].plot(kind='bar', ax=axes[3], color='red')
    axes[3].set_title('Maximum Drawdown')
    axes[3].set_ylabel('Drawdown (%)')
    axes[3].grid(axis='y', alpha=0.3)

    # Plot VaR
    df.loc['VaR (95%)'].plot(kind='bar', ax=axes[4], color='purple')
    axes[4].set_title('Value at Risk (95%)')
    axes[4].set_ylabel('VaR (%)')
    axes[4].grid(axis='y', alpha=0.3)

    # Hide the unused subplot
    axes[5].set_visible(False)

    # Add value labels on top of bars
    for ax in axes[:5]:
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}%',
                      (p.get_x() + p.get_width() / 2., p.get_height()),
                      ha='center', va='bottom')

    plt.tight_layout()
    fig.suptitle('Risk and Return Metrics Comparison', fontsize=16, y=1.02)

    # Save the plot if output_file provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved risk metrics plot to {output_file}")

    return fig


def main(weights_file: str,
          returns_file: str,
          output_dir: str,
          benchmark_files: Optional[Dict[str, str]] = None,
          risk_free_rate: float = 0.02) -> None:
    """
    Main function to generate portfolio visualization.

    Parameters:
    -----------
    weights_file : str
        Path to the CSV file containing portfolio weights
    returns_file : str
        Path to the CSV file containing returns data
    output_dir : str
        Directory to save the output files
    benchmark_files : Optional[Dict[str, str]], optional
        Dictionary of benchmark name and file paths, by default None
    risk_free_rate : float, optional
        Risk-free rate (annualized), by default 0.02
    """
    print(f"Generating portfolio visualizations")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load portfolio weights and returns
    weights, returns_data = load_data(weights_file, returns_file)
    if weights is None or returns_data is None:
        return

    # Calculate portfolio returns
    if 'portfolio_return' in returns_data.columns:
        portfolio_returns = returns_data['portfolio_return']
    else:
        # Filter returns to include only assets in weights
        common_assets = set(weights.index).intersection(set(returns_data.columns))
        filtered_weights = weights[list(common_assets)]
        filtered_returns = returns_data[list(common_assets)]

        # Normalize weights to sum to 1
        filtered_weights = filtered_weights / filtered_weights.sum()

        # Calculate portfolio returns
        portfolio_returns = (filtered_returns * filtered_weights).sum(axis=1)

    # Calculate performance metrics
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

    performance_metrics = {
        'annualized_return': annualized_return * 100,  # Convert to percentage
        'annualized_volatility': annualized_volatility * 100,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown * 100,
        'var_95': var_95 * 100
    }

    # 1. Plot portfolio weights
    weights_plot_file = os.path.join(output_dir, 'portfolio_weights.png')
    plot_portfolio_weights(weights, output_file=weights_plot_file)

    # 2. Plot rolling returns histogram
    histogram_file = os.path.join(output_dir, 'rolling_returns_histogram.png')
    plot_rolling_returns_histogram(portfolio_returns, output_file=histogram_file)

    # 3. Load benchmark returns if provided
    benchmark_returns = pd.DataFrame()
    benchmark_metrics = {}

    if benchmark_files:
        for name, file_path in benchmark_files.items():
            try:
                bench_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
                if name in bench_data.columns:
                    benchmark_returns[name] = bench_data[name]
                else:
                    # Assume first column is the returns
                    benchmark_returns[name] = bench_data.iloc[:, 0]

                # Calculate benchmark metrics
                bench_returns = benchmark_returns[name]
                bench_ann_return = (1 + bench_returns.mean()) ** 252 - 1
                bench_volatility = bench_returns.std() * np.sqrt(252)
                bench_sharpe = (bench_ann_return - risk_free_rate) / bench_volatility

                # Maximum drawdown
                bench_cum_returns = (1 + bench_returns).cumprod()
                bench_running_max = bench_cum_returns.cummax()
                bench_drawdown = (bench_cum_returns / bench_running_max) - 1
                bench_max_drawdown = bench_drawdown.min()

                # Value at Risk (95%)
                bench_var_95 = bench_returns.quantile(0.05)

                benchmark_metrics[name] = {
                    'annualized_return': bench_ann_return * 100,
                    'annualized_volatility': bench_volatility * 100,
                    'sharpe_ratio': bench_sharpe,
                    'max_drawdown': bench_max_drawdown * 100,
                    'var_95': bench_var_95 * 100
                }

                print(f"Loaded benchmark: {name}")
            except Exception as e:
                print(f"Error loading benchmark {name}: {e}")

    # 4. Plot benchmark comparison if benchmarks provided
    if not benchmark_returns.empty:
        comparison_file = os.path.join(output_dir, 'benchmark_comparison.png')
        plot_benchmark_comparison(portfolio_returns, benchmark_returns, output_file=comparison_file)

        # 5. Plot risk metrics comparison
        metrics_file = os.path.join(output_dir, 'risk_metrics.png')
        plot_risk_metrics(performance_metrics, benchmark_metrics, output_file=metrics_file)

    print(f"\nVisualization complete. Output saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Portfolio visualization")
    parser.add_argument("--weights", "-w", required=True, help="Path to the weights CSV file")
    parser.add_argument("--returns", "-r", required=True, help="Path to the returns CSV file")
    parser.add_argument("--output", "-o", required=True, help="Directory to save output files")
    parser.add_argument("--risk-free-rate", type=float, default=0.02,
                        help="Risk-free rate (annualized)")
    parser.add_argument("--spy-benchmark", help="Path to S&P 500 (SPY) returns CSV")
    parser.add_argument("--bond-benchmark", help="Path to 60/40 portfolio returns CSV")

    args = parser.parse_args()

    # Set up benchmark files dictionary if provided
    benchmark_files = {}
    if args.spy_benchmark:
        benchmark_files['SPY'] = args.spy_benchmark
    if args.bond_benchmark:
        benchmark_files['60/40 Portfolio'] = args.bond_benchmark

    main(
        args.weights,
        args.returns,
        args.output,
        benchmark_files=benchmark_files if benchmark_files else None,
        risk_free_rate=args.risk_free_rate
    )
