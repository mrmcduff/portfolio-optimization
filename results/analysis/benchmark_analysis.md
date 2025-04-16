# Benchmark Analysis Report

**Analysis Period:** 2018-06-20 to 2025-01-15 (1653 trading days, ~6.6 years)

## Performance Metrics

|                       | S&P 500   | 60/40 Portfolio   | Custom Algorithm Portfolio   |
|:----------------------|:----------|:------------------|:-----------------------------|
| Total Return          | 138.65%   | 78.22%            | 129.83%                      |
| Annualized Return     | 16.41%    | 10.04%            | 20.65%                       |
| Annualized Volatility | 19.62%    | 12.25%            | 20.47%                       |
| Sharpe Ratio          | 0.734     | 0.656             | 0.911                        |
| Jensen's Alpha        | -0.01%    | -0.47%            | 3.16%                        |
| Beta                  | 1.00      | 0.61              | 0.93                         |
| Max Drawdown          | -33.72%   | -21.19%           | -32.49%                      |
| Value at Risk (95%)   | -1.85%    | -1.12%            | -1.82%                       |
| Conditional VaR (95%) | -2.99%    | -1.84%            | -3.13%                       |
| Skewness              | -0.530    | -0.665            | -0.530                       |
| Kurtosis              | 11.954    | 14.681            | 9.952                        |

## Return Correlations

|                            |   S&P 500 |   60/40 Portfolio |   Custom Algorithm Portfolio |
|:---------------------------|----------:|------------------:|-----------------------------:|
| S&P 500                    |  1        |          0.977159 |                     0.950754 |
| 60/40 Portfolio            |  0.977159 |          1        |                     0.919913 |
| Custom Algorithm Portfolio |  0.950754 |          0.919913 |                     1        |

## Benchmark Descriptions

### S&P 500

Market benchmark used for calculating Beta and Jensen's Alpha.

### 60/40 Portfolio

Portfolio or benchmark returns series.

### Custom Algorithm Portfolio

Portfolio or benchmark returns series.

## Notes

- Sharpe Ratio assumes a risk-free rate of 0%.
- Max Drawdown represents the largest peak-to-trough decline during the period.
- Value at Risk (95%) indicates the worst expected loss over a day with 95% confidence.
- Conditional VaR (95%) represents the average loss on days when losses exceed the 95% VaR.
- Jensen's Alpha measures the portfolio's excess return relative to what would be predicted by CAPM.
- Beta measures the portfolio's systematic risk relative to the market benchmark.
- Market benchmark used: S&P 500
