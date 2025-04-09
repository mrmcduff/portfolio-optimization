"""
ETF Data Preprocessing Module

This script processes raw ETF price data and prepares it for portfolio optimization.
It generates returns, correlation matrices, and summary statistics.
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Import period configurations
from src.config.periods import PERIOD_RANGES


def load_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load ETF price data from CSV file.

    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing the ETF price data

    Returns:
    --------
    Optional[pd.DataFrame]
        Loaded data or None if an error occurs
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Successfully loaded data from {file_path}")
        print(f"Columns: {data.columns.tolist()}")
        print(f"Rows: {len(data)}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the ETF price data.

    Parameters:
    -----------
    data : pd.DataFrame
        Raw ETF price data

    Returns:
    --------
    pd.DataFrame
        Preprocessed data with dates as index
    """
    # Make a copy of the data
    df = data.copy()

    # Convert Date column to datetime
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        # Set Date as index
        df.set_index("Date", inplace=True)
        # Sort by date
        df = df.sort_index()

    print(f"Preprocessed data: {df.shape}")
    return df


def calculate_returns(prices: pd.DataFrame, method: str = "simple") -> pd.DataFrame:
    """
    Calculate returns from price data.

    Parameters:
    -----------
    prices : pd.DataFrame
        Price data with dates as index
    method : str, optional
        Method to calculate returns: 'simple' or 'log', by default 'simple'

    Returns:
    --------
    pd.DataFrame
        Returns data
    """
    if method == "log":
        returns = np.log(prices / prices.shift(1))
    else:  # simple returns
        returns = prices.pct_change()

    # Drop rows with NaN values
    returns = returns.dropna()

    print(f"Calculated {method} returns: {returns.shape}")
    return returns


def resample_returns(returns: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    """
    Resample returns to a different frequency.

    Parameters:
    -----------
    returns : pd.DataFrame
        Returns data
    freq : str, optional
        Frequency to resample to: 'D' (daily), 'W' (weekly),
        'M' (monthly), 'Q' (quarterly), by default 'M'

    Returns:
    --------
    pd.DataFrame
        Resampled returns
    """
    if freq == "D":
        return returns  # Already daily

    # For other frequencies, compound the returns
    return returns.resample(freq).apply(lambda x: (1 + x).prod() - 1)


def calculate_statistics(
    returns: pd.DataFrame, periods_per_year: int = 252
) -> pd.DataFrame:
    """
    Calculate summary statistics for returns.

    Parameters:
    -----------
    returns : pd.DataFrame
        Returns data
    periods_per_year : int, optional
        Number of periods per year for annualization, by default 252 (daily)

    Returns:
    --------
    pd.DataFrame
        Summary statistics
    """
    # Mean return (annualized)
    mean_return = returns.mean() * periods_per_year

    # Volatility (annualized)
    volatility = returns.std() * np.sqrt(periods_per_year)

    # Sharpe ratio (assuming 0% risk-free rate for simplicity)
    sharpe_ratio = mean_return / volatility

    # Maximum drawdown
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / running_max) - 1
    max_drawdown = drawdown.min()

    # Skewness and kurtosis
    skewness = returns.skew()
    kurtosis = returns.kurtosis()

    # Compile statistics
    stats = pd.DataFrame(
        {
            "annualized_return": mean_return,
            "annualized_volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "skewness": skewness,
            "kurtosis": kurtosis,
        }
    )

    print("Calculated summary statistics")
    return stats


def split_by_period(
    returns: pd.DataFrame, periods: Dict[str, Tuple[str, str]]
) -> Dict[str, pd.DataFrame]:
    """
    Split returns data by specified periods.

    Parameters:
    -----------
    returns : pd.DataFrame
        Returns data
    periods : Dict[str, Tuple[str, str]]
        Dictionary with period names as keys and (start_date, end_date) tuples as values

    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary with period names as keys and filtered returns as values
    """
    period_returns: Dict[str, pd.DataFrame] = {}

    for name, (start, end) in periods.items():
        # Convert string dates to datetime objects for proper comparison
        start_date = pd.to_datetime(start)
        end_date = pd.to_datetime(end)

        # Filter data using datetime objects
        mask = (returns.index >= start_date) & (returns.index <= end_date)
        period_returns[name] = returns.loc[mask]
        print(f"Period {name}: {start} to {end}, rows: {len(period_returns[name])}")

    return period_returns


def prepare_sector_data(
    returns: pd.DataFrame, period: Optional[str] = None
) -> pd.DataFrame:
    """
    Prepare sector ETF data for portfolio optimization.

    Parameters:
    -----------
    returns : pd.DataFrame
        Returns data for all ETFs
    period : Optional[str], optional
        Period name to prepare data for, by default None
        If specified, will handle ETFs that didn't exist during certain periods

    Returns:
    --------
    pd.DataFrame
        Returns data for sector ETFs only
    """
    # List of sector ETFs
    sector_etfs: List[str] = [col for col in returns.columns if col.startswith("XL")]

    # Filter returns data to only include the selected ETFs
    sector_returns = returns[sector_etfs]

    # Drop rows with any NaN values
    sector_returns = sector_returns.dropna()

    print(
        f"Extracted {len(sector_etfs)} sector ETFs for {period or 'all periods'}: {sector_etfs}"
    )
    print(f"Final data shape: {sector_returns.shape}")
    return sector_returns


def prepare_financial_crisis_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data specifically for the financial crisis period.
    This function processes the raw data directly to avoid losing early data.

    Parameters:
    -----------
    raw_data : pd.DataFrame
        Raw price data with dates as index

    Returns:
    --------
    pd.DataFrame
        Returns data for the financial crisis period
    """
    print("\nPreparing specialized financial crisis data...")

    # Make a copy of the data
    df = raw_data.copy()

    # Filter to financial crisis period
    start_date, end_date = PERIOD_RANGES["financial_crisis"]
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    # Filter by date
    mask = (df.index >= start_dt) & (df.index <= end_dt)
    crisis_data = df.loc[mask]

    print(
        f"Financial crisis period data: {start_date} to {end_date}, rows: {len(crisis_data)}"
    )

    # Find sector ETFs that have data during this period
    sector_etfs = [col for col in crisis_data.columns if col.startswith("XL")]
    valid_etfs = []

    for etf in sector_etfs:
        # Check if this ETF has enough non-NaN values in the period (at least 80%)
        valid_count = crisis_data[etf].count()
        total_count = len(crisis_data)
        if valid_count / total_count >= 0.8:  # At least 80% of data points are valid
            valid_etfs.append(etf)
        else:
            print(
                f"Excluding {etf} from financial crisis analysis: only {valid_count}/{total_count} valid data points"
            )

    # Filter to valid ETFs only
    crisis_data = crisis_data[valid_etfs]

    # Calculate returns
    crisis_returns = crisis_data.pct_change().dropna()

    print(f"Using {len(valid_etfs)} sector ETFs for financial crisis: {valid_etfs}")
    print(f"Final financial crisis returns data shape: {crisis_returns.shape}")

    return crisis_returns


def save_processed_data(
    data_dict: Dict[str, pd.DataFrame], output_dir: str
) -> List[str]:
    """
    Save processed data to CSV files.

    Parameters:
    -----------
    data_dict : Dict[str, pd.DataFrame]
        Dictionary with data names as keys and DataFrames as values
    output_dir : str
        Directory to save the files

    Returns:
    --------
    List[str]
        List of saved file paths
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    saved_files: List[str] = []
    for name, df in data_dict.items():
        # Create a valid filename
        filename = f"{name.replace(' ', '_').lower()}.csv"
        file_path = os.path.join(output_dir, filename)

        # Save to CSV
        df.to_csv(file_path)
        saved_files.append(file_path)
        print(f"Saved {name} to {file_path}")

    return saved_files


def main(input_file: str, output_dir: str) -> None:
    """
    Main function to process ETF price data.

    Parameters:
    -----------
    input_file : str
        Path to the input CSV file
    output_dir : str
        Directory to save the processed data
    """
    print(f"Processing {input_file}")

    # Load data
    data = load_data(input_file)
    if data is None:
        return

    # Preprocess data
    prices = preprocess_data(data)

    # Calculate returns
    returns = calculate_returns(prices)

    # Define full period based on actual data availability
    min_date = returns.index.min()
    max_date = returns.index.max()

    # Use standard periods from config but add full_period
    periods = PERIOD_RANGES.copy()
    periods["full_period"] = (
        min_date.strftime("%Y-%m-%d"),
        max_date.strftime("%Y-%m-%d"),
    )

    print("Using standard period definitions:")
    for period_name, (start, end) in periods.items():
        print(f"  {period_name}: {start} to {end}")

    # Split returns by period
    period_returns = split_by_period(returns, periods)

    # Calculate statistics for each period
    period_stats = {
        f"{name}_stats": calculate_statistics(period_data)
        for name, period_data in period_returns.items()
    }

    # Prepare sector ETF data for all periods
    sector_returns = prepare_sector_data(returns)

    # Prepare specialized data for the financial crisis period directly from raw data
    # This ensures we don't lose early data due to NaN values in newer ETFs
    financial_crisis_returns = prepare_financial_crisis_data(prices)

    # Calculate correlation and covariance matrices for the recent period
    recent_returns = period_returns["recent"]
    correlation = recent_returns.corr()
    covariance = recent_returns.cov() * 252  # Annualized

    # Resample returns to different frequencies
    monthly_returns = resample_returns(returns, freq="M")
    quarterly_returns = resample_returns(returns, freq="Q")

    # Prepare data for the fast algorithm
    fa_inputs = pd.DataFrame(
        {
            "expected_return": recent_returns.mean() * 252,  # Annualized
            "volatility": recent_returns.std() * np.sqrt(252),  # Annualized
        }
    )

    # Save all processed data
    data_to_save = {
        "daily_returns": returns,
        "monthly_returns": monthly_returns,
        "quarterly_returns": quarterly_returns,
        "sector_returns": sector_returns,
        "financial_crisis_sector_returns": financial_crisis_returns,
        "recent_returns": recent_returns,
        "correlation_matrix": correlation,
        "covariance_matrix": covariance,
        "fa_inputs": fa_inputs,
    }

    # Add period statistics
    data_to_save.update(period_stats)

    # Save all data
    saved_files = save_processed_data(data_to_save, output_dir)

    print(f"\nProcessing complete. {len(saved_files)} files saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process ETF price data for portfolio optimization"
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Path to the input CSV file"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Directory to save the processed data"
    )

    args = parser.parse_args()

    main(args.input, args.output)
