import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd


class SingleIndexModel:
    """
    Implementation of the Sharpe Single-Index Model for portfolio optimization.

    Parameters
    ----------
    returns : pd.DataFrame
        Security returns
    market_returns : pd.Series
        Market returns
    risk_free_rate : float
        Annualized risk-free rate
    output_dir : str
        Directory to save logs and results
    logger : callable or object, optional
        Logger function or object for logging output. If None, logs to 'ofa_algorithm_log.txt'.
        If verbose is True, logs to both file and console.
    verbose : bool, optional
        If True, logs will be printed to console as well as file.
    """

    """
    Implementation of the Sharpe Single-Index Model for portfolio optimization.
    """

    # Define sector ETFs at the class level for consistent use across methods
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

    def __init__(
        self,
        returns: pd.DataFrame,
        market_returns: pd.Series,
        risk_free_rate: float = 0.02,
        output_dir: str = "results/models",
        logger=None,
        verbose: bool = False,
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
        output_dir : str, optional
            Directory to save logs and results, by default "results/models"
        logger : callable or object, optional
            Logger function or object for logging output. If None, logs to 'ofa_algorithm_log.txt'.
            If verbose is True, logs to both file and console.
        verbose : bool, optional
            If True, logs will be printed to console as well as file.
        """
        # Returns are already in decimal form
        self.returns = returns
        self.market_returns = market_returns
        self.risk_free_rate = risk_free_rate
        self.parameters = {}
        self.verbose = verbose
        self.output_dir = output_dir
        self.output_file = os.path.join(output_dir, "ofa_selection_analysis.csv")
        os.makedirs(output_dir, exist_ok=True)
        # Setup logger
        if logger is None:
            log_path = os.path.join(output_dir, "ofa_algorithm_log.txt")
            # Delete log file at the start of each run to ensure a clean slate
            if os.path.exists(log_path):
                try:
                    os.remove(log_path)
                except Exception as e:
                    print(f"Warning: Could not delete log file {log_path}: {e}")

            def default_logger(msg):
                with open(log_path, "a") as f:
                    f.write(msg + "\n")
                if verbose:
                    print(msg)

            self.logger = default_logger
        else:
            self.logger = logger
        # Only clear the file if it doesn't exist or is empty
        if (
            not os.path.exists(self.output_file)
            or os.path.getsize(self.output_file) == 0
        ):
            with open(self.output_file, "w") as f:
                f.write("")  # Create empty file
            self.logger(f"\nCleared selection analysis file: {self.output_file}")
        # Initialize selection log for this period
        self.selection_log = []
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
            self.logger(f"\nSelection analysis appended to {self.output_file}")
        except Exception:
            pass
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
        self.logger("\nEstimating parameters for each security:")

        # Calculate market parameters safely
        if isinstance(self.market_returns.var(), pd.Series):
            market_variance = float(self.market_returns.var().iloc[0])
        else:
            market_variance = float(self.market_returns.var())

        if market_variance == 0 or np.isnan(market_variance):
            raise ValueError(
                "Market variance is zero or NaN, cannot estimate parameters"
            )

        # Calculate daily risk-free rate
        daily_rf = (1 + self.risk_free_rate) ** (1 / 252) - 1

        self.parameters = {}
        current_date = self.returns.index[-1]

        # First pass: calculate basic parameters for each security
        for security in self.returns.columns:
            if security not in self.X_ITEMS:
                continue
            try:
                y = self.returns[security]
                x = (
                    self.market_returns.iloc[:, 0]
                    if isinstance(self.market_returns, pd.DataFrame)
                    else self.market_returns
                )
                # Calculate beta using covariance-based formula: Beta_i = Cov(R_i, R_m) / Var(R_m)
                beta = float(np.cov(y, x)[0, 1] / np.var(x)) if np.var(x) > 0 else 0

                # Calculate alpha: alpha = E[R_i] - beta * E[R_m]
                mean_return = float(
                    y.mean().iloc[0] if isinstance(y.mean(), pd.Series) else y.mean()
                )
                alpha = mean_return - beta * x.mean()
                excess_return = mean_return - daily_rf
                # Calculate residual variance with a minimum threshold
                # Calculate e_i_t = R_i_t - (alpha_i + beta_i*R_m_t) for each observation
                residuals = y - (alpha + beta * x)

                # Variance of the residuals
                residual_variance = max(
                    float(residuals.var()), 1e-6
                )  # Minimum to avoid division by zero

                # Calculate excess return to beta ratio
                excess_return_to_beta = (
                    excess_return / beta if beta > 0 else float("-inf")
                )

                excess_return_times_beta_over_residual_var = (
                    excess_return * beta / residual_variance
                    if beta > 0
                    else float("-inf")
                )

                beta_sq_over_residual_var = (
                    beta**2 / residual_variance if beta > 0 else float("-inf")
                )

                # Store parameters
                self.parameters[security] = {
                    "alpha": alpha,
                    "beta": beta,
                    "residual_variance": residual_variance,
                    "excess_return_to_beta": excess_return_to_beta,
                    "mean_return": mean_return,
                    "excess_return": excess_return,
                    "excess_return_times_beta_over_residual_var": excess_return_times_beta_over_residual_var,
                    "beta_sq_over_residual_var": beta_sq_over_residual_var,
                }

                # Create a selection log entry
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
                        "Excess Return*Beta/Residual Var": excess_return_times_beta_over_residual_var,
                        "Beta^2/Residual Var": beta_sq_over_residual_var,
                        "Selected": False,  # Will be updated during portfolio calculation
                    }
                )
            except Exception as e:
                self.logger(
                    f"[WARNING] Skipping security '{security}' in parameter estimation due to error: {e}"
                )

        # Second pass: sort by ER/Beta and calculate C_i values
        er_beta_list = [
            (s, self.parameters[s]["excess_return_to_beta"])
            for s in self.returns.columns
            if s in self.parameters
        ]
        sorted_by_er_beta = sorted(er_beta_list, key=lambda x: x[1], reverse=True)
        er_beta_ranks = {s: i + 1 for i, (s, _) in enumerate(sorted_by_er_beta)}

        # Calculate sum terms for c_i calculation
        # Running sums for each security in order of descending ER/Beta
        sum_of_er_b_over_resvar = 0
        sum_of_beta_sq_over_resvar = 0

        # Iterate through securities in descending ER/Beta order
        for security, _ in sorted_by_er_beta:
            if security in self.parameters:
                # Get the precomputed values
                er_b_over_resvar = self.parameters[security][
                    "excess_return_times_beta_over_residual_var"
                ]
                beta_sq_over_resvar = self.parameters[security][
                    "beta_sq_over_residual_var"
                ]

                # Update running sums
                sum_of_er_b_over_resvar += er_b_over_resvar
                sum_of_beta_sq_over_resvar += beta_sq_over_resvar

                # Store the running sums in the parameters
                self.parameters[security][
                    "sum_of_er_b_over_resvar"
                ] = sum_of_er_b_over_resvar
                self.parameters[security][
                    "sum_of_beta_sq_over_resvar"
                ] = sum_of_beta_sq_over_resvar

        # Calculate C_i values for each security using the formula c_i = (M*S_i) / (1+ M*R_i)
        c_values = {}

        # M = market variance
        M = market_variance

        # Print headers for the sorted table
        self.logger("-" * 100)
        self.logger(
            f"{'Security':<10} {'Beta':>10} {'Alpha':>10} {'Mean Ret':>10} {'Exc Ret':>10} {'Res Var':>10} {'ER/Beta':>10} {'C_i':>10} {'Selected':>10}"
        )
        self.logger("-" * 100)

        # Process securities in order of descending ER/Beta
        for security, _ in sorted_by_er_beta:
            params = self.parameters[security]
            beta = params["beta"]
            alpha = params["alpha"]
            mean_return = params["mean_return"]
            excess_return = params["excess_return"]
            residual_variance = params["residual_variance"]
            er_beta = params["excess_return_to_beta"]

            # Get the running sums S_i and R_i for this security
            S_i = params["sum_of_er_b_over_resvar"]
            R_i = params["sum_of_beta_sq_over_resvar"]

            try:
                # For negative beta, we don't want to include in c_i calculation
                if beta <= 0:
                    c_i = 0  # For negative beta, C_i is set to 0
                else:
                    # Calculate C_i using the formula: c_i = (M*S_i) / (1+ M*R_i)
                    c_i = (M * S_i) / (1 + M * R_i) if (1 + M * R_i) != 0 else 0

                c_values[security] = c_i
            except Exception:
                # Handle potential errors (division by zero, etc.)
                c_i = 0
                c_values[security] = 0

            # Determine if security would be selected (ER/Beta > C_i)
            # A security is selected if its ER/Beta is greater than its C_i value
            selected = "Yes" if er_beta > c_i else "No"

            # Print in sorted order with C_i and Selected columns
            self.logger(
                f"{security:<10} {beta:>10.4f} {alpha:>10.4f} {mean_return:>10.4f} "
                f"{excess_return:>10.4f} {residual_variance:>10.4f} {er_beta:>10.4f} "
                f"{c_i:>10.4f} {selected:>10}"
            )

            # Update selection log with C_i values
            for entry in self.selection_log:
                if entry["Security"] == security and entry["Date"] == current_date:
                    entry["C_i"] = c_i
                    entry["Rank"] = er_beta_ranks.get(security, 0)

        # Store C_i values for later use
        self.c_values = c_values

        self.logger(
            f"\nEstimated parameters successfully for {len(self.parameters)} securities."
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

        # Calculate market parameters
        market_mean = float(
            self.market_returns.mean().iloc[0]
            if isinstance(self.market_returns.mean(), pd.Series)
            else self.market_returns.mean()
        )
        market_variance = float(
            self.market_returns.var().iloc[0]
            if isinstance(self.market_returns.var(), pd.Series)
            else self.market_returns.var()
        )
        daily_rf = (1 + self.risk_free_rate) ** (1 / 252) - 1

        # Market parameters for reference
        market_excess_return = market_mean - daily_rf

        self.logger("Market Parameters:")
        self.logger(f"Market Mean Return: {market_mean:.6f}")
        self.logger(f"Market Variance: {market_variance:.6f}")
        self.logger(f"Daily Risk-Free Rate: {daily_rf:.6f}")
        self.logger(f"Market Excess Return: {market_excess_return:.6f}")

        current_date = self.returns.index[-1]

        # Calculate optimal weights
        weights = {}
        c_values = {}  # Store C_i and Z_i values
        z_values = {}  # Store Z_i values for logging

        # Compute ER/Beta ranks and sort securities
        er_beta_list = [
            (s, self.parameters[s]["excess_return_to_beta"])
            for s in self.returns.columns
            if s in self.parameters and s in self.X_ITEMS
        ]
        sorted_by_er_beta = sorted(er_beta_list, key=lambda x: x[1], reverse=True)
        sorted_securities = [s for s, _ in sorted_by_er_beta]
        er_beta_ranks = {s: i + 1 for i, (s, _) in enumerate(sorted_by_er_beta)}

        # First pass: calculate all C_i values
        sum_numerator = 0
        sum_denominator = 0

        # Process securities in order of descending ER/Beta

        for security in sorted_securities:
            params = self.parameters[security]
            beta = params["beta"]
            residual_variance = params["residual_variance"]
            excess_return = params["excess_return"]
            alpha = params["alpha"]
            er_beta = params["excess_return_to_beta"]

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

        # Calculate C* using the improved approach where we compare ER/Beta against C_i BEFORE including a security
        sum_numerator = 0
        sum_denominator = 0
        c_star = 0.0
        selected_securities = []
        # Store the selection C_i value for each security (the C_i value used to make selection decision)
        selection_c_i_values = {}

        # Process securities in descending ER/Beta order
        for security in sorted_securities:
            params = self.parameters[security]
            beta = params["beta"]
            residual_variance = params["residual_variance"]
            excess_return = params["excess_return"]
            er_beta = params["excess_return_to_beta"]

            # Special case for the first security (highest ER/Beta)
            # The first security should always be selected if its ER/Beta > 0
            if len(selected_securities) == 0 and er_beta > 0:
                # For the first security, set C_i so it's guaranteed to be selected
                # Take 80% of ER/Beta to ensure selection
                current_c_i = er_beta * 0.8 if er_beta > 0 else 0
                selection_c_i_values[security] = current_c_i
                selected = True
            else:
                # Normal case: Get the current C_i before adding this security
                current_c_i = (
                    sum_numerator / sum_denominator if sum_denominator > 0 else 0
                )
                selection_c_i_values[security] = current_c_i
                # Select if ER/Beta > current C_i
                selected = er_beta > current_c_i

            # If selected, add to the list
            if selected:
                selected_securities.append(security)

                # Update sums for next C_i calculation
                term = (beta / residual_variance) * excess_return / beta
                sum_numerator += term
                sum_denominator += beta**2 / residual_variance
            else:
                # No more securities to include
                break

        # Final C* is the last C_i calculated after adding all selected securities
        c_star = sum_numerator / sum_denominator if sum_denominator > 0 else 0

        self.logger(f"C* (Cut-off Rate): {c_star:.6f}")

        # Calculate Z_i values and weights using the selected securities and found C*
        weights = {}

        # Calculate Z_i values and weights
        # Z_i is ONLY calculated for securities where ER/Beta > C*
        # All other securities have Z_i = 0
        self.logger(
            f"{'Security':<10} {'ER/Beta':>10} {'C_i':>10} {'Z_i':>10} {'Weight':>10} {'Selected':>10}"
        )

        for security in sorted_securities:
            params = self.parameters[security]
            beta = params["beta"]
            residual_variance = params["residual_variance"]
            excess_return = params["excess_return"]
            er_beta = params["excess_return_to_beta"]
            c_i = c_values.get(security, 0)

            # Only calculate Z_i for securities in the selected list
            # These are the securities where ER/Beta > C* (selection criterion)
            if security in selected_securities:
                try:
                    # IMPORTANT: Use the selection_c_i value that was used to make the decision
                    # NOT the final c_star value
                    selection_c_i = selection_c_i_values.get(security, 0)

                    # By definition, er_beta > selection_c_i for selected securities
                    z_i = (beta / residual_variance) * (er_beta - selection_c_i)

                    # This should now be positive by definition of selection
                    weights[security] = max(0, z_i)  # Use max as a safety
                except Exception:
                    z_i = 0.0
                    weights[security] = 0.0
            else:
                # For unselected securities, Z_i is 0 by definition
                z_i = 0.0
                weights[security] = 0.0

            # Record the Z_i value
            z_values[security] = z_i

            # Flag to show which securities were selected
            is_selected = "Yes" if security in selected_securities else "No"

            self.logger(
                f"{security:<10} {er_beta:>10.4f} {c_i:>10.4f} {z_i:>10.4f} {weights[security]:>10.4f} {is_selected:>10}"
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
                    entry["Rank"] = er_beta_ranks.get(
                        security, 0
                    )  # Use the actual rank from er_beta_ranks

        # Filter out zero weights
        # Only keep positive weights for X-items
        weights = {k: v for k, v in weights.items() if v > 0 and k in self.X_ITEMS}

        # Normalize remaining weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
            self.logger("\nFinal normalized weights:")
            for security, weight in weights.items():
                self.logger(f"{security:<10} {weight:>10.4f}")
                # Update selection log to mark selected securities (only X-items)
                for entry in self.selection_log:
                    if entry["Security"] == security and entry["Date"] == current_date:
                        entry["Selected"] = True
        else:
            self.logger("\nFalling back to equal weights (X-items only)")
            # If all calculations fail, equal weight the X-items only
            n_securities = len(self.X_ITEMS)
            weight = 1.0 / n_securities
            weights = {security: weight for security in self.X_ITEMS}
            # Update selection log to mark all X-items as selected
            for entry in self.selection_log:
                if entry["Security"] in self.X_ITEMS and entry["Date"] == current_date:
                    entry["Selected"] = True
                    entry["Final Weight"] = weight

        # Save portfolio weights and expected return for logging

        self.current_portfolio_weights_json = json.dumps(weights)
        self.current_portfolio_expected_return = sum(
            weights.get(s, 0) * self.parameters[s]["mean_return"]
            for s in weights
            if s in self.parameters and s in self.X_ITEMS
        )

        # Calculate realized return for the next period if available
        realized_return = None
        date_idx = list(self.returns.index).index(current_date)
        if date_idx + 1 < len(self.returns.index):
            next_date = self.returns.index[date_idx + 1]
            realized_return = sum(
                weights.get(s, 0) * self.returns.loc[next_date, s]
                for s in weights
                if s in self.returns.columns
                and s in self.parameters
                and s in self.X_ITEMS
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
        ranked_securities = sorted(
            [
                s
                for s in self.returns.columns
                if s in self.parameters and s in self.X_ITEMS
            ],
            key=lambda s: self.parameters[s]["excess_return_to_beta"],
            reverse=True,
        )

        # We use the lowest ranked security's ratio as C*
        lowest_security = ranked_securities[-1]
        c_star = self.parameters[lowest_security]["excess_return_to_beta"]
        # Track C* for logging
        self.current_c_star = c_star

        # Calculate Z_i for all securities (short sales allowed)
        z_values = {}
        for s in ranked_securities:
            param = self.parameters[s]
            z_i = (
                (param["excess_return"] - c_star)
                * param["beta"]
                / param["residual_variance"]
            )
            z_values[s] = z_i

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
            weights[s] * self.returns[s].mean() for s in self.X_ITEMS if s in weights
        )

        # Calculate portfolio beta
        portfolio_beta = sum(
            weights[s] * self.parameters[s]["beta"]
            for s in self.X_ITEMS
            if s in weights
        )

        # Calculate residual variance contribution
        residual_variance_contribution = sum(
            (weights[s] ** 2) * self.parameters[s]["residual_variance"]
            for s in self.X_ITEMS
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

    # Generate correlated returns for sector ETFs (X_ITEMS)
    securities_returns = {}
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
    for i, ticker in enumerate(X_ITEMS):
        beta = 0.8 + 0.1 * i  # Betas from 0.8 upwards
        alpha = 0.0005 * (5 - i % 11)  # Some variation in alphas
        idiosyncratic = np.random.normal(0, 0.01 * (1 + (i % 3)), 30)
        returns = alpha + beta * market_returns + idiosyncratic
        securities_returns[ticker] = returns

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
    model.logger("Estimated Parameters:")
    for security, param in params.items():
        model.logger(f"{security}:")
        model.logger(f"  Alpha: {param['alpha']:.6f}")
        model.logger(f"  Beta: {param['beta']:.6f}")
        model.logger(f"  Residual Variance: {param['residual_variance']:.6f}")
        model.logger(f"  Excess Return / Beta: {param['excess_return_to_beta']:.6f}")
        model.logger("")

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
