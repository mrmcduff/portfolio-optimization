# Benchmark Analysis Report

**Analysis Period:** 2018-06-20 to 2025-01-15 (1653 trading days, ~6.6 years)

## Performance Metrics

|                          |   Total Return |   Annualized Return |   Annualized Volatility |   Sharpe Ratio |   Max Drawdown |   Value at Risk (95%) |   Skewness |   Kurtosis |
|:-------------------------|---------------:|--------------------:|------------------------:|---------------:|---------------:|----------------------:|-----------:|-----------:|
| S&P 500                  |       1.38652  |            0.164062 |                0.196151 |       0.734443 |      -0.337173 |            -0.0184559 |  -0.530175 |    11.9544 |
| 60/40 Portfolio          |       0.782242 |            0.100394 |                0.122469 |       0.656448 |      -0.211871 |            -0.0112389 |  -0.665305 |    14.6811 |
| Fast Algorithm Portfolio |       1.14077  |            0.187318 |                0.195723 |       0.854871 |      -0.330251 |            -0.0164847 |  -0.525451 |    15.1121 |

## Return Correlations

|                          |   S&P 500 |   60/40 Portfolio |   Fast Algorithm Portfolio |
|:-------------------------|----------:|------------------:|---------------------------:|
| S&P 500                  |  1        |          0.977159 |                   0.977352 |
| 60/40 Portfolio          |  0.977159 |          1        |                   0.94943  |
| Fast Algorithm Portfolio |  0.977352 |          0.94943  |                   1        |

## Benchmark Descriptions

### S&P 500

The S&P 500 Index is a market-capitalization-weighted index of the 500 largest publicly traded companies in the U.S. It is widely regarded as the best gauge of large-cap U.S. equities.

### 60/40 Portfolio

A traditional balanced portfolio consisting of 60% stocks (S&P 500) and 40% bonds. This allocation is considered a benchmark for moderate investors seeking growth with some downside protection.

### Fast Algorithm Portfolio

A portfolio optimized using the Fast Algorithm implementation of the Markowitz Mean-Variance Optimization framework. This portfolio seeks to maximize the Sharpe ratio by finding the optimal allocation across assets.

## Notes

- Sharpe Ratio assumes a risk-free rate of 0%.
- Max Drawdown represents the largest peak-to-trough decline during the period.
- Value at Risk (95%) indicates the worst expected loss over a day with 95% confidence.
