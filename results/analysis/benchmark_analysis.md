# Benchmark Analysis Report

**Analysis Period:** 2018-06-20 to 2025-01-15 (1653 trading days, ~6.6 years)

## Performance Metrics

|                       | S&P 500   | 60/40 Portfolio   | Returns Algorithm Portfolio   | Weighted Top Five Portfolio   |
|:----------------------|:----------|:------------------|:------------------------------|:------------------------------|
| Total Return          | 138.65%   | 78.22%            | 149.46%                       | 146.00%                       |
| Annualized Return     | 16.41%    | 10.04%            | 17.01%                        | 17.01%                        |
| Annualized Volatility | 19.62%    | 12.25%            | 18.77%                        | 19.88%                        |
| Sharpe Ratio          | 0.734     | 0.656             | 0.800                         | 0.755                         |
| Jensen's Alpha        | -0.01%    | -0.47%            | 2.35%                         | 2.34%                         |
| Beta                  | 1.00      | 0.61              | 0.86                          | 0.87                          |
| Max Drawdown          | -33.72%   | -21.19%           | -34.53%                       | -32.51%                       |
| Value at Risk (95%)   | -1.85%    | -1.12%            | -1.73%                        | -1.83%                        |
| Conditional VaR (95%) | -2.99%    | -1.84%            | -2.78%                        | -2.92%                        |
| Skewness              | -0.530    | -0.665            | -0.518                        | -0.326                        |
| Kurtosis              | 11.954    | 14.681            | 17.183                        | 12.371                        |

## Return Correlations

|                             |   S&P 500 |   60/40 Portfolio |   Returns Algorithm Portfolio |   Weighted Top Five Portfolio |
|:----------------------------|----------:|------------------:|------------------------------:|------------------------------:|
| S&P 500                     |  1        |          0.977159 |                      0.903633 |                      0.853499 |
| 60/40 Portfolio             |  0.977159 |          1        |                      0.88418  |                      0.834318 |
| Returns Algorithm Portfolio |  0.903633 |          0.88418  |                      1        |                      0.97565  |
| Weighted Top Five Portfolio |  0.853499 |          0.834318 |                      0.97565  |                      1        |

## Benchmark Descriptions

### S&P 500

Market benchmark used for calculating Beta and Jensen's Alpha.

### 60/40 Portfolio

Portfolio or benchmark returns series.

### Returns Algorithm Portfolio

Portfolio or benchmark returns series.

### Weighted Top Five Portfolio

Portfolio or benchmark returns series.

## Notes

- Sharpe Ratio assumes a risk-free rate of 0%.
- Max Drawdown represents the largest peak-to-trough decline during the period.
- Value at Risk (95%) indicates the worst expected loss over a day with 95% confidence.
- Conditional VaR (95%) represents the average loss on days when losses exceed the 95% VaR.
- Jensen's Alpha measures the portfolio's excess return relative to what would be predicted by CAPM.
- Beta measures the portfolio's systematic risk relative to the market benchmark.
- Market benchmark used: S&P 500
