import os

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = "results/models"
FIGURES_DIR = "results/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# File paths for the four portfolios
spy_file = os.path.join(RESULTS_DIR, "spy_returns.csv")
balanced_file = os.path.join(RESULTS_DIR, "balanced_returns_M.csv")
custom_file = os.path.join(RESULTS_DIR, "cust_portfolio_returns.csv")
fast_file = os.path.join(RESULTS_DIR, "ofa_portfolio_returns.csv")

# Portfolio display names
portfolio_names = [
    (spy_file, "S&P 500"),
    (balanced_file, "60/40 Portfolio"),
    (custom_file, "Custom Algorithm"),
    (fast_file, "One Factor Fast Algorithm"),
]

# Load returns (assume daily returns, index as date)
returns = []
labels = []
for fpath, label in portfolio_names:
    df = pd.read_csv(fpath, index_col=0, parse_dates=True)
    # Use the first column if multiple columns
    if df.shape[1] > 1:
        df = df.iloc[:, 0]
    else:
        df = df.squeeze()
    returns.append(df)
    labels.append(label)

returns_df = pd.concat(returns, axis=1)
returns_df.columns = labels
returns_df = returns_df.dropna(how="all")

# Plot cumulative returns
cum_returns = (1 + returns_df).cumprod()
plt.figure(figsize=(14, 7))
for col in cum_returns.columns:
    plt.plot(cum_returns.index, cum_returns[col], label=col, linewidth=2)
plt.title("Cumulative Returns Comparison")
plt.xlabel("Date")
plt.ylabel("Growth of $1")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(FIGURES_DIR, "returns_comparison.png"), dpi=300, bbox_inches="tight"
)
plt.close()

# Plot rolling Sharpe ratios (252-day window)
risk_free_rate = 0.05 / 252  # daily risk-free rate
window = 252
sharpe_df = pd.DataFrame(index=returns_df.index)
for col in returns_df.columns:
    excess = returns_df[col] - risk_free_rate
    rolling_mean = excess.rolling(window).mean()
    rolling_std = returns_df[col].rolling(window).std()
    sharpe = rolling_mean / rolling_std
    sharpe_df[col] = sharpe

plt.figure(figsize=(14, 7))
for col in sharpe_df.columns:
    plt.plot(sharpe_df.index, sharpe_df[col], label=col, linewidth=2)
plt.title(f"Rolling {window}-Day Sharpe Ratio Comparison")
plt.xlabel("Date")
plt.ylabel("Sharpe Ratio")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(FIGURES_DIR, "sharpe_comparison.png"), dpi=300, bbox_inches="tight"
)
plt.close()
