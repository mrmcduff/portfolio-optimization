# Rebalancing Strategy Comparison

Analysis period: 2018-06-20 to 2025-01-15

## Performance Metrics

|                             | Total Return   | Annualized Return   | Annualized Volatility   |   Sharpe Ratio | Max Drawdown   | VaR (95%)   | CVaR (95%)   |
|:----------------------------|:---------------|:--------------------|:------------------------|---------------:|:---------------|:------------|:-------------|
| S&P 500                     | 138.65%        | 16.41%              | 19.62%                  |       0.734443 | -33.72%        | -1.85%      | -2.99%       |
| 60/40 No Rebalancing        | 79.78%         | 10.21%              | 12.45%                  |       0.65915  | -21.80%        | -1.11%      | -1.86%       |
| 60/40 Monthly Rebalancing   | 78.22%         | 10.04%              | 12.25%                  |       0.656448 | -21.19%        | -1.12%      | -1.84%       |
| 60/40 Quarterly Rebalancing | 79.19%         | 10.14%              | 12.33%                  |       0.660428 | -21.03%        | -1.13%      | -1.85%       |
| 60/40 Threshold Rebalancing | 79.93%         | 10.23%              | 12.44%                  |       0.661321 | -21.89%        | -1.13%      | -1.86%       |

## Understanding Rebalancing Strategies

### No Rebalancing
Portfolio weights drift with market movements, which typically increases equity exposure over time in bull markets.

### Monthly Rebalancing
Portfolio is rebalanced to target weights at the end of each calendar month, maintaining consistent risk exposure.

### Quarterly Rebalancing
Portfolio is rebalanced every three months, balancing transaction costs with risk control.

### Threshold Rebalancing
Portfolio is rebalanced only when allocations drift more than 5% from targets, potentially reducing unnecessary transactions.

## Analysis Insights

- **Best Return**: S&P 500 with 16.41% annualized
- **Best Risk-Adjusted Return**: S&P 500 with Sharpe ratio of 0.73
- **Lowest Drawdown**: 60/40 Quarterly Rebalancing with maximum drawdown of -21.03%

## Rebalancing Implications

During trending markets, less frequent rebalancing or no rebalancing may capture more upside from the better-performing asset class. However, this comes at the cost of increased risk from higher concentration.

More frequent rebalancing provides better risk control and more consistent exposure to the investor's target allocation. This approach may underperform during strong trending markets but offers better protection during periods of mean reversion.

Threshold-based rebalancing attempts to balance these concerns by only trading when allocations drift significantly, potentially reducing transaction costs while still providing reasonable risk control.
