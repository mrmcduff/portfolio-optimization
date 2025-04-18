# Custom Algorithm for Portfolio Optimization

This document explains the Custom Algorithm implementation for portfolio optimization used in our system. The algorithm is based on Modern Portfolio Theory (MPT) and uses mathematical optimization techniques to find the optimal asset allocation.

## Overview

The Custom Algorithm is designed to find the optimal portfolio allocation that maximizes the Sharpe ratio (risk-adjusted return) or achieves a target return with minimum risk. It uses quadratic programming techniques to solve the optimization problem.

## Step-by-Step Process

### 1. Data Preparation

The process begins with historical price data for assets (in our case, sector ETFs):

1. **Load Returns Data**:
   - Load historical returns from CSV files
   - Special handling for specific periods (e.g., financial crisis period uses a specialized dataset that excludes ETFs that didn't exist during that time)

2. **Filter by Period**:
   - Filter returns data to the specified time period (financial_crisis, post_crisis, or recent)
   - Check if there's enough data after filtering

3. **Calculate Expected Returns and Covariance Matrix**:
   ```python
   # Expected returns (annualized)
   expected_returns = returns.mean() * periods_per_year

   # Covariance matrix (annualized)
   covariance_matrix = returns.cov() * periods_per_year
   ```
   - The expected returns are calculated as the mean of historical returns, annualized by multiplying by the number of periods per year (252 for daily data)
   - The covariance matrix captures the relationships between asset returns and is also annualized

### 2. Portfolio Optimization

The core optimization function (`custom_algorithm_portfolio`) implements the following steps:

1. **Define the Objective Function**:
   - If maximizing Sharpe ratio (default):
     ```python
     def objective(weights):
         portfolio_return = np.sum(returns * weights)
         portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
         sharpe = (portfolio_return - risk_free_rate) / portfolio_volatility
         return -sharpe  # Negative because we're minimizing
     ```
   - If targeting a specific return:
     ```python
     def objective(weights):
         portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
         return portfolio_variance  # Minimize variance
     ```

2. **Define Constraints**:
   - **Budget Constraint**: Weights must sum to 1 (100% allocation)
     ```python
     constraints.append({"type": "eq", "fun": lambda x: np.sum(x) - 1})
     ```
   - **Return Constraint** (if target_return is specified):
     ```python
     constraints.append({"type": "eq", "fun": lambda x: np.sum(returns * x) - target_return})
     ```

3. **Set Bounds for Weights**:
   - **Long-only Constraint** (if enabled): Weights must be non-negative
   - **Maximum Weight Constraint** (if specified): No asset can exceed the maximum weight
     ```python
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
     ```

4. **Run Optimization**:
   - Use Sequential Least Squares Programming (SLSQP) method to solve the optimization problem
   - Start with an equal-weight portfolio as the initial guess
     ```python
     result = minimize(
         objective, init_guess, method="SLSQP", bounds=bounds, constraints=constraints
     )
     ```

5. **Post-Optimization Adjustments**:
   - Apply minimum weight constraint (set weights below min_weight to zero)
   - Rescale remaining weights to sum to 1
   - Apply maximum weight constraint after rescaling
   - Redistribute excess weight from capped assets to other assets
   - Iterate until constraints are satisfied

6. **Calculate Portfolio Metrics**:
   - Expected return: Weighted sum of asset expected returns
   - Volatility: Square root of the portfolio variance
   - Sharpe ratio: (Portfolio return - Risk-free rate) / Portfolio volatility

### 3. Output and Analysis

After optimization, the algorithm:

1. **Saves the Results**:
   - Optimal weights for each asset
   - Portfolio metrics (return, volatility, Sharpe ratio)
   - Portfolio returns time series
   - Rolling annual returns for performance analysis

2. **Displays Portfolio Information**:
   - Expected return
   - Expected volatility
   - Sharpe ratio
   - Optimal asset allocation

## Key Parameters

The algorithm accepts several parameters that affect the optimization:

- **risk_free_rate**: The risk-free rate used in Sharpe ratio calculation (default: 0.02 or 2%)
- **long_only**: Whether to enforce long-only constraint (default: True)
- **max_weight**: Maximum weight for any asset (default: 0.25 or 25%)
- **min_weight**: Minimum weight for any asset if included (default: 0.01 or 1%)
- **period**: Time period to analyze (financial_crisis, post_crisis, recent)

## Mathematical Foundation

The Custom Algorithm is based on Modern Portfolio Theory, which seeks to maximize expected return for a given level of risk, or minimize risk for a given level of expected return.

The optimization problem can be formulated as:

For maximizing Sharpe ratio:
- Maximize: (R_p - R_f) / σ_p
- Subject to: Σw_i = 1
- And: 0 ≤ w_i ≤ max_weight (for long-only portfolios)

Where:
- R_p is the portfolio expected return
- R_f is the risk-free rate
- σ_p is the portfolio standard deviation (volatility)
- w_i is the weight of asset i

For minimum variance at a target return:
- Minimize: w^T Σ w
- Subject to: w^T μ = target_return
- And: Σw_i = 1
- And: 0 ≤ w_i ≤ max_weight (for long-only portfolios)

Where:
- w is the vector of weights
- Σ is the covariance matrix
- μ is the vector of expected returns

## Practical Considerations

- **Maximum Weight Constraint**: Setting a maximum weight (e.g., 25%) ensures diversification but may lead to perfectly even allocations when multiple assets hit this constraint
- **Minimum Weight Constraint**: Eliminates very small positions that might be impractical to implement
- **Long-only Constraint**: Prevents short selling, which may not be allowed in certain investment contexts

## Conclusion

The Custom Algorithm provides a robust method for portfolio optimization based on historical returns data. By adjusting the parameters, investors can tailor the optimization to their specific investment constraints and objectives.
