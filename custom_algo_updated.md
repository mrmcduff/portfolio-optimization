# Custom Algorithm: Long-Only Selection and Rebalancing Procedure

This document explains the complete selection algorithm implemented for the custom (long-only) portfolio, including the logic applied at each rebalancing period.

---

## 1. Overview
At each rebalancing date, the algorithm selects portfolio weights using a mean-variance optimization framework (the "Fast Algorithm"). The optimization is subject to long-only constraints and user-defined bounds on minimum and maximum weights per asset. The process is repeated at each rebalancing interval (e.g., quarterly), using a rolling lookback window of historical returns for parameter estimation.

---

## 2. Detailed Step-by-Step Algorithm

### **Step 1: Data Preparation**
- **Input:** Historical returns data for all candidate assets, sorted by date.
- **Lookback Window:** For each rebalancing date, select the most recent `lookback_window` days of returns (e.g., 252 trading days).

### **Step 2: Parameter Estimation**
- **Expected Returns:** Compute mean returns for each asset over the lookback window, annualized.
- **Covariance Matrix:** Compute the sample covariance matrix of returns over the lookback window, annualized.

### **Step 3: Portfolio Optimization (Long-Only)**
- **Objective:**
    - If no target return is specified: maximize Sharpe ratio (tangency portfolio).
    - If a target return is specified: minimize portfolio variance subject to achieving the target return.
- **Constraints:**
    - **Budget:** All portfolio weights sum to 1.
    - **Long-Only:** All weights are constrained to be ≥ 0.
    - **Max Weight:** No asset weight exceeds a specified maximum (e.g., 25%).
    - **Min Weight:** Any asset included must have a weight ≥ specified minimum (e.g., 1%).
- **Bounds:** Set accordingly for each asset: [min_weight, max_weight] (or [0, max_weight] if no min_weight).
- **Solver:** Use Sequential Least Squares Programming (SLSQP) to solve the constrained optimization.

### **Step 4: Post-Optimization Adjustment**
- Iteratively enforce min and max weight constraints:
    - Set weights below min_weight to zero, then rescale remaining weights to sum to 1.
    - Cap weights above max_weight, then distribute any excess to non-maxed weights, rescaling as needed.
    - Repeat until all constraints are satisfied or a maximum number of iterations is reached.

### **Step 5: Portfolio Construction**
- The resulting weights define the portfolio for the next period.
- If optimization fails at any rebalancing, fall back to equal-weighted allocation.

### **Step 6: Portfolio Return Calculation**
- For each day until the next rebalance, calculate portfolio returns using the current weights and actual asset returns.
- At the next rebalancing date, repeat the process with updated data.

---

## 3. Pseudocode Summary
```python
for each rebalancing_date in rebalancing_dates:
    # 1. Select lookback window data
    lookback_data = returns.loc[lookback_start:lookback_end]

    # 2. Estimate expected returns and covariance
    expected_returns = lookback_data.mean() * periods_per_year
    cov_matrix = lookback_data.cov() * periods_per_year

    # 3. Optimize weights (long-only, weight bounds)
    optimal_weights = optimize_portfolio(
        expected_returns,
        cov_matrix,
        risk_free_rate=risk_free_rate,
        long_only=True,
        max_weight=max_weight,
        min_weight=min_weight,
    )

    # 4. Post-process weights to satisfy min/max constraints
    # (Iterative adjustment and rescaling as described above)

    # 5. Apply weights for next period's returns
    ...
```

---

## 4. Key Features
- **Long-Only:** No short positions are allowed; all weights are ≥ 0.
- **Dynamic Rebalancing:** Portfolio is re-optimized at each rebalancing date using the most recent data.
- **Robust to Optimization Failures:** Falls back to equal weights if optimization fails at any rebalance.
- **Transparent Selection:** The selection and weighting process is fully determined by historical risk/return and user constraints.

---

## 5. References
- Markowitz, H. (1952). Portfolio Selection. Journal of Finance.
- Documentation and implementation in `src/optimization/custom_algorithm.py` (see `fast_algorithm_portfolio` and `analyze_portfolio_performance_with_rebalancing`).
