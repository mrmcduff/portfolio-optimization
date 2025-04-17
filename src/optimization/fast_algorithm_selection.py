import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd


class SingleIndexModel:
    """
    Implementation of the Sharpe Single-Index Model for portfolio optimization.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        market_returns: pd.Series,
        risk_free_rate: float = 0.02,
    ):
        """
        Initialize the Single Index Model.

        Parameters:
        -----------
        returns : pd.DataFrame
            DataFrame of security returns (in decimal form)
        market_returns : pd.Series
            Series of market returns (in decimal form)
        risk_free_rate : float, optional
            Annualized risk-free rate, by default 0.02
        """
        # Returns are already in decimal form
        self.returns = returns
        self.market_returns = market_returns
        self.risk_free_rate = risk_free_rate
        self.parameters = {}
        self._validate_data()

        # Initialize selection log for this period
        self.selection_log = []

        # Prepare output file path
        output_dir = "results/models"
        self.output_file = os.path.join(output_dir, "ofa_selection_analysis.csv")
        os.makedirs(output_dir, exist_ok=True)
        # Only clear the file if it doesn't exist or is empty
        if (
            not os.path.exists(self.output_file)
            or os.path.getsize(self.output_file) == 0
        ):
            with open(self.output_file, "w") as f:
                f.write("")  # Create empty file
            print(f"\nCleared selection analysis file: {self.output_file}")
        # Initialize current_c_star for logging
        self.current_c_star = None

    def _validate_data(self):
        """Validate input data and handle missing values."""
        # Ensure market returns and security returns have the same index
        common_index = self.returns.index.intersection(self.market_returns.index)
        if len(common_index) < 2:
            raise ValueError(
                "Insufficient overlapping data between market and security returns"
            )

        self.returns = self.returns.loc[common_index]
        self.market_returns = self.market_returns.loc[common_index]

        # Handle missing values using the new methods
        self.returns = self.returns.ffill().bfill()
        self.market_returns = self.market_returns.ffill().bfill()

    def _save_selection_analysis(self):
        """Save or append selection analysis to CSV file."""
        if not self.selection_log:
            return

        # Add C* and other debug info to each row in the log
        for entry in self.selection_log:
            entry["C*"] = getattr(self, "current_c_star", None)
            entry["Portfolio Expected Return"] = getattr(
                self, "current_portfolio_expected_return", None
            )
            entry["Portfolio Realized Return"] = getattr(
                self, "current_portfolio_realized_return", None
            )
            entry["Portfolio Weights"] = getattr(
                self, "current_portfolio_weights_json", None
            )
            entry["Rebalance Date"] = entry.get("Date", None)
        current_log = pd.DataFrame(self.selection_log)

        try:
            # Append new rows to the CSV file (do not overwrite)
            header = (
                not os.path.exists(self.output_file)
                or os.path.getsize(self.output_file) == 0
            )
            current_log.to_csv(self.output_file, mode="a", header=header, index=False)
            print(f"\nSelection analysis appended to {self.output_file}")
        except Exception as e:
            print(f"Warning: Could not save selection analysis: {e}")
        # Clear selection_log after saving to avoid duplicate rows
        self.selection_log.clear()

    def estimate_parameters(self) -> Dict[str, Dict[str, float]]:
        """
        Estimate parameters for each security using the Single Index Model.

        Returns:
        --------
        Dict[str, Dict[str, float]]
            Dictionary of parameters for each security
        """
        print("\nEstimating parameters for each security:")
        print("-" * 80)
        print(
            f"{'Security':<10} {'Beta':>10} {'Alpha':>10} {'Mean Ret':>10} {'Exc Ret':>10} {'Res Var':>10} {'ER/Beta':>10}"
        )
        print("-" * 80)

        # Calculate market parameters
        market_mean = float(self.market_returns.mean())
        market_variance = float(self.market_returns.var())

        if market_variance == 0 or np.isnan(market_variance):
            raise ValueError(
                "Market variance is zero or NaN, cannot estimate parameters"
            )

        # Calculate daily risk-free rate
        daily_rf = (1 + self.risk_free_rate) ** (1 / 252) - 1

        # Calculate market excess returns
        market_excess_returns = self.market_returns - daily_rf

        self.parameters = {}
        current_date = self.returns.index[-1]

        # Only use X-items (sector ETFs) for all calculations
        X_ITEMS = [
            "XLB",
            "XLC",
            "XLE",
            "XLF",
            "XLI",
            "XLK",
            "XLP",
            "XLRE",
            "XLU",
            "XLV",
            "XLY",
        ]
        for security in self.returns.columns:
            if security not in X_ITEMS:
                continue
            try:
                y = self.returns[security]
                x = (
                    self.market_returns.iloc[:, 0]
                    if isinstance(self.market_returns, pd.DataFrame)
                    else self.market_returns
                )
                # Linear regression: y = alpha + beta * x
                beta, alpha = np.polyfit(x, y, 1)
                mean_return = float(y.mean())
                excess_return = mean_return - daily_rf
                # Calculate residual variance with a minimum threshold
                residual_variance = max(
                    float((y - (alpha + beta * x)).var()),
                    1e-6,  # Small positive number to avoid division by zero
                )
                # Calculate excess return to beta ratio
                try:
                    excess_return_to_beta = (
                        excess_return / beta if beta > 0 else float("-inf")
                    )
                except Exception:
                    excess_return_to_beta = float("-inf")
                self.parameters[security] = {
                    "alpha": alpha,
                    "beta": beta,
                    "residual_variance": residual_variance,
                    "excess_return_to_beta": excess_return_to_beta,
                    "mean_return": mean_return,
                    "excess_return": excess_return,
                }
                print(
                    f"{security:<10} {beta:>10.4f} {alpha:>10.4f} {mean_return:>10.4f} {excess_return:>10.4f} {residual_variance:>10.4f} {excess_return_to_beta:>10.4f}"
                )
                self.selection_log.append(
                    {
                        "Date": current_date,
                        "Security": security,
                        "Beta": beta,
                        "Alpha": alpha,
                        "Mean Return": mean_return,
                        "Excess Return": excess_return,
                        "Residual Variance": residual_variance,
                        "Excess Return/Beta": excess_return_to_beta,
                        "Selected": False,
                    }
                )
            except Exception as e:
                print(
                    f"[WARNING] Skipping security '{security}' in parameter estimation due to error: {e}"
                )

        # Compute ER/Beta ranks for logging only for securities with parameters
        er_beta_list = [
            (s, self.parameters[s]["excess_return_to_beta"])
            for s in self.returns.columns
            if s in self.parameters
        ]
        sorted_by_er_beta = sorted(er_beta_list, key=lambda x: x[1], reverse=True)
        er_beta_ranks = {s: i + 1 for i, (s, _) in enumerate(sorted_by_er_beta)}

        return self.parameters

        print("\nEstimating parameters for each security:")
        print("-" * 80)
        print(
            f"{'Security':<10} {'Beta':>10} {'Alpha':>10} {'Mean Ret':>10} {'Exc Ret':>10} {'Res Var':>10} {'ER/Beta':>10}"
        )
        print("-" * 80)

        # Calculate market parameters
        market_mean = float(self.market_returns.mean())
        market_variance = float(self.market_returns.var())

        if market_variance == 0 or np.isnan(market_variance):
            raise ValueError(
                "Market variance is zero or NaN, cannot estimate parameters"
            )

        # Calculate daily risk-free rate
        daily_rf = (1 + self.risk_free_rate) ** (1 / 252) - 1

        # Calculate market excess returns
        market_excess_returns = self.market_returns - daily_rf

        current_date = self.returns.index[-1]  # Get the most recent date

        for security in self.returns.columns:
            # Calculate security excess returns
            security_returns = self.returns[security]
            mean_return = float(security_returns.mean())
            excess_returns = security_returns - daily_rf

            # Calculate beta using covariance
            try:
                covariance = float(np.cov(excess_returns, market_excess_returns)[0, 1])
                beta = covariance / market_variance
            except:
                beta = 1.0  # Default to market beta if calculation fails

            # Calculate alpha
            alpha = float(excess_returns.mean() - beta * market_excess_returns.mean())

            # Calculate residual variance with a minimum threshold
            residual_variance = max(
                float(excess_returns.var() - (beta**2 * market_variance)),
                1e-6,  # Small positive number to avoid division by zero
            )

            # Calculate excess return to beta ratio
            excess_return = mean_return - daily_rf
            try:
                excess_return_to_beta = (
                    excess_return / beta if beta > 0 else float("-inf")
                )
            except:
                excess_return_to_beta = float("-inf")

            # Store parameters
            self.parameters[security] = {
                "alpha": alpha,
                "beta": beta,
                "residual_variance": residual_variance,
                "excess_return_to_beta": excess_return_to_beta,
                "mean_return": mean_return,
                "excess_return": excess_return,
            }

            # Print parameter values
            print(
                f"{security:<10} {beta:>10.4f} {alpha:>10.4f} {mean_return:>10.4f} {excess_return:>10.4f} {residual_variance:>10.4f} {excess_return_to_beta:>10.4f}"
            )

            # Store selection data
            self.selection_log.append(
                {
                    "Date": current_date,
                    "Security": security,
                    "Beta": beta,
                    "Alpha": alpha,
                    "Mean Return": mean_return,
                    "Excess Return": excess_return,
                    "Residual Variance": residual_variance,
                    "Excess Return/Beta": excess_return_to_beta,
                    "Selected": False,  # Will be updated during portfolio calculation
                }
            )

        return self.parameters

    def calculate_optimal_portfolio(self) -> Dict[str, float]:
        """
        Calculate optimal portfolio weights using the Single Index Model.

        Returns:
        --------
        Dict[str, float]
            Dictionary of optimal weights for each security
        """
        if not self.parameters:
            self.estimate_parameters()

        print("\nCalculating optimal portfolio weights:")
        print("-" * 80)

        # Calculate market parameters
        market_mean = float(self.market_returns.mean())
        market_variance = float(self.market_returns.var())
        daily_rf = (1 + self.risk_free_rate) ** (1 / 252) - 1

        # Calculate c* (market risk premium)
        market_excess_return = market_mean - daily_rf
        c_star = market_excess_return / market_variance if market_variance > 0 else 0

        print("Market Parameters:")
        print(f"Market Mean Return: {market_mean:.6f}")
        print(f"Market Variance: {market_variance:.6f}")
        print(f"Daily Risk-Free Rate: {daily_rf:.6f}")
        print(f"Market Excess Return: {market_excess_return:.6f}")
        print(f"C* (Cut-off Rate): {c_star:.6f}")
        print("-" * 80)
        print(f"{'Security':<10} {'Z_i':>10} {'C_i':>10} {'Weight':>10}")
        print("-" * 80)

        current_date = self.returns.index[-1]

        # Calculate optimal weights
        weights = {}
        c_values = {}  # Store C_i values for each security
        z_values = {}  # Store Z_i values for logging

        # Compute ER/Beta ranks for logging
        er_beta_list = [
            (s, self.parameters[s]["excess_return_to_beta"])
            for s in self.returns.columns
            if s in self.parameters
        ]
        sorted_by_er_beta = sorted(er_beta_list, key=lambda x: x[1], reverse=True)
        er_beta_ranks = {s: i + 1 for i, (s, _) in enumerate(sorted_by_er_beta)}

        # First pass: calculate all C_i values
        sum_numerator = 0
        sum_denominator = 0

        import json

        for security, params in self.parameters.items():
            beta = params["beta"]
            residual_variance = params["residual_variance"]
            excess_return = params["excess_return"]
            alpha = params["alpha"]
            er_beta = params["excess_return_to_beta"]
            rank = er_beta_ranks.get(security, None)

            try:
                term = (beta / residual_variance) * excess_return / beta
                sum_numerator += term
                sum_denominator += beta**2 / residual_variance
                # Calculate C_i for this security
                c_i = sum_numerator / sum_denominator if sum_denominator > 0 else 0
                c_values[security] = c_i
            except Exception:
                c_i = 0
                c_values[security] = 0

        # Second pass: calculate Z_i values using C* and weights
        for security, params in self.parameters.items():
            beta = params["beta"]
            residual_variance = params["residual_variance"]
            excess_return = params["excess_return"]
            alpha = params["alpha"]
            er_beta = params["excess_return_to_beta"]
            rank = er_beta_ranks.get(security, None)
            c_i = c_values[security]

            # Calculate z_i with error handling
            try:
                z_i = (beta / residual_variance) * (excess_return / beta - c_star)
                weights[security] = max(0, z_i)  # Only keep positive weights
            except Exception:
                z_i = 0.0
                weights[security] = 0.0
            z_values[security] = z_i

            print(
                f"{security:<10} {z_i:>10.4f} {c_i:>10.4f} {weights[security]:>10.4f}"
            )

            # Update the selection log entry for this security with all debug info
            for entry in self.selection_log:
                if entry["Security"] == security and entry["Date"] == current_date:
                    entry["C_i"] = c_i
                    entry["Z_i"] = z_i
                    entry["Beta"] = beta
                    entry["Alpha"] = alpha
                    entry["Residual Variance"] = residual_variance
                    entry["Excess Return"] = excess_return
                    entry["Excess Return/Beta"] = er_beta
                    entry["Rank"] = rank

        # Filter out zero weights
        # Only keep positive weights for X-items
        X_ITEMS = [
            "XLB",
            "XLC",
            "XLE",
            "XLF",
            "XLI",
            "XLK",
            "XLP",
            "XLRE",
            "XLU",
            "XLV",
            "XLY",
        ]
        weights = {k: v for k, v in weights.items() if v > 0 and k in X_ITEMS}

        # Normalize remaining weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
            print("\nFinal normalized weights:")
            for security, weight in weights.items():
                print(f"{security:<10} {weight:>10.4f}")
                # Update selection log to mark selected securities (only X-items)
                for entry in self.selection_log:
                    if entry["Security"] == security and entry["Date"] == current_date:
                        entry["Selected"] = True
        else:
            print("\nFalling back to equal weights (X-items only)")
            # If all calculations fail, equal weight the X-items only
            n_securities = len(X_ITEMS)
            weight = 1.0 / n_securities
            weights = {security: weight for security in X_ITEMS}
            # Update selection log to mark all X-items as selected
            for entry in self.selection_log:
                if entry["Security"] in X_ITEMS and entry["Date"] == current_date:
                    entry["Selected"] = True
                    entry["Final Weight"] = weight

        # Save portfolio weights and expected return for logging

        self.current_portfolio_weights_json = json.dumps(weights)
        self.current_portfolio_expected_return = sum(
            weights.get(s, 0) * self.parameters[s]["mean_return"]
            for s in weights
            if s in self.parameters and s in X_ITEMS
        )

        # Calculate realized return for the next period if available
        realized_return = None
        date_idx = list(self.returns.index).index(current_date)
        if date_idx + 1 < len(self.returns.index):
            next_date = self.returns.index[date_idx + 1]
            realized_return = sum(
                weights.get(s, 0) * self.returns.loc[next_date, s]
                for s in weights
                if s in self.returns.columns and s in self.parameters and s in X_ITEMS
            )
        self.current_portfolio_realized_return = realized_return

        # Update selection log with portfolio metrics
        for entry in self.selection_log:
            if entry["Date"] == current_date:
                entry[
                    "Portfolio Expected Return"
                ] = self.current_portfolio_expected_return
                entry[
                    "Portfolio Realized Return"
                ] = self.current_portfolio_realized_return
                entry["Portfolio Weights"] = self.current_portfolio_weights_json

        # Save selection log to CSV
        self._save_selection_analysis()

        return weights

    def optimize_with_short_sales(
        self, lintner_method: bool = False
    ) -> Tuple[Dict[str, float], Dict]:
        """
        Calculate the optimal portfolio with short sales allowed.

        Parameters:
        -----------
        lintner_method : bool
            If True, uses Lintner's method for scaling weights. If False, uses traditional method.

        Returns:
        --------
        Tuple[Dict[str, float], Dict]:
            - Dictionary mapping securities to their optimal weights
            - Dictionary with portfolio statistics
        """
        # Make sure parameters are estimated
        if not self.parameters:
            self.estimate_parameters()

        # Rank securities by excess return to beta ratio (descending)
        X_ITEMS = [
            "XLB",
            "XLC",
            "XLE",
            "XLF",
            "XLI",
            "XLK",
            "XLP",
            "XLRE",
            "XLU",
            "XLV",
            "XLY",
        ]
        ranked_securities = sorted(
            [s for s in self.returns.columns if s in self.parameters and s in X_ITEMS],
            key=lambda s: self.parameters[s]["excess_return_to_beta"],
            reverse=True,
        )

        # We use the lowest ranked security's ratio as C*
        lowest_security = ranked_securities[-1]
        c_star = self.parameters[lowest_security]["excess_return_to_beta"]
        # Track C* for logging
        self.current_c_star = c_star

        if lintner_method:
            sum_abs_z = sum(abs(z) for z in z_values.values())
            weights = {security: z_i / sum_abs_z for security, z_i in z_values.items()}
        else:
            sum_z = sum(z_values.values())
            weights = {security: z_i / sum_z for security, z_i in z_values.items()}

        # Calculate portfolio statistics
        market_variance = self.market_returns.var(ddof=0)

        # Calculate expected portfolio return and variance
        portfolio_return = sum(
            weights[s] * self.returns[s].mean() for s in X_ITEMS if s in weights
        )

        # Calculate portfolio beta
        portfolio_beta = sum(
            weights[s] * self.parameters[s]["beta"] for s in X_ITEMS if s in weights
        )

        # Calculate residual variance contribution
        residual_variance_contribution = sum(
            (weights[s] ** 2) * self.parameters[s]["residual_variance"]
            for s in X_ITEMS
            if s in weights
        )

        # Calculate portfolio variance
        portfolio_variance = (
            portfolio_beta**2
        ) * market_variance + residual_variance_contribution

        # Calculate portfolio standard deviation
        portfolio_std = np.sqrt(portfolio_variance)

        # Return results
        portfolio_stats = {
            "expected_return": portfolio_return,
            "variance": portfolio_variance,
            "std_dev": portfolio_std,
            "beta": portfolio_beta,
            "sharpe_ratio": (portfolio_return - self.risk_free_rate) / portfolio_std,
            "c_star": c_star,
            "method": "Lintner" if lintner_method else "Traditional",
        }

        # Save selection analysis with C* for this run
        self._save_selection_analysis()
        return weights, portfolio_stats


# Example usage
def example_usage():
    # Create a sample DataFrame with returns
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=30, freq="D")

    # Generate random returns for SPY (market) and 5 securities
    market_returns = np.random.normal(0.001, 0.02, 30)  # Mean of 0.1%, std of 2%

    # Generate correlated returns for securities
    securities_returns = {}
    for i in range(1, 6):
        # Generate returns with different betas and alphas
        beta = 0.5 + i * 0.3  # Betas from 0.8 to 2.0
        alpha = 0.0005 * (3 - i)  # Alphas from 0.001 to -0.001
        idiosyncratic = np.random.normal(
            0, 0.01 * i, 30
        )  # Different idiosyncratic volatility
        returns = alpha + beta * market_returns + idiosyncratic
        securities_returns[f"Stock_{i}"] = returns

    # Create DataFrame
    securities_returns["SPY"] = market_returns
    returns_df = pd.DataFrame(securities_returns, index=dates)

    # Initialize and run the model
    model = SingleIndexModel(
        returns_df, market_returns=returns_df["SPY"], risk_free_rate=0.0003
    )  # 0.03% daily risk-free rate

    # Estimate parameters
    params = model.estimate_parameters()

    # Print parameters
    print("Estimated Parameters:")
    for security, param in params.items():
        print(f"{security}:")
        print(f"  Alpha: {param['alpha']:.6f}")
        print(f"  Beta: {param['beta']:.6f}")
        print(f"  Residual Variance: {param['residual_variance']:.6f}")
        print(f"  Excess Return / Beta: {param['excess_return_to_beta']:.6f}")
        print()

    # Calculate optimal portfolio (long-only)
    weights = model.calculate_optimal_portfolio()

    print("Optimal Portfolio (No Short Sales):")
    print("Weights:")
    for security, weight in weights.items():
        print(f"  {security}: {weight:.4f}")

    # Calculate optimal portfolio with short sales
    weights_short, stats_short = model.optimize_with_short_sales(lintner_method=False)

    print("Optimal Portfolio (With Short Sales - Traditional Method):")
    print("Weights:")
    for security, weight in weights_short.items():
        print(f"  {security}: {weight:.4f}")

    # Calculate optimal portfolio with short sales (Lintner method)
    weights_lintner, stats_lintner = model.optimize_with_short_sales(
        lintner_method=True
    )

    print("\nOptimal Portfolio (With Short Sales - Lintner Method):")
    print("Weights:")
    for security, weight in weights_lintner.items():
        print(f"  {security}: {weight:.4f}")

    print("\nPortfolio Statistics:")
    print(f"  Expected Return: {stats_lintner['expected_return']:.6f}")
    print(f"  Standard Deviation: {stats_lintner['std_dev']:.6f}")
    print(f"  Sharpe Ratio: {stats_lintner['sharpe_ratio']:.6f}")
    print(f"  Beta: {stats_lintner['beta']:.6f}")


if __name__ == "__main__":
    example_usage()
