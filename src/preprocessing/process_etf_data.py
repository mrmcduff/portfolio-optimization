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
        mask = (returns.index >= start) & (returns.index <= end)
        period_returns[name] = returns.loc[mask]
        print(f"Period {name}: {start} to {end}, rows: {len(period_returns[name])}")

    return period_returns


def prepare_sector_data(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare sector ETF data for portfolio optimization.

    Parameters:
    -----------
    returns : pd.DataFrame
        Returns data for all ETFs

    Returns:
    --------
    pd.DataFrame
        Returns data for sector ETFs only
    """
    # List of sector ETFs
    sector_etfs: List[str] = [col for col in returns.columns if col.startswith("XL")]

    # Filter returns data
    sector_returns = returns[sector_etfs]

    print(f"Extracted {len(sector_etfs)} sector ETFs: {sector_etfs}")
    return sector_returns


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

    # Define periods for analysis
    periods = {
        "full_period": (returns.index.min(), returns.index.max()),
        "financial_crisis": ("2007-01-01", "2011-12-31"),
        "post_crisis": ("2012-01-01", "2018-12-31"),
        "recent": ("2019-01-01", "2023-12-31"),
    }

    # Split returns by period
    period_returns = split_by_period(returns, periods)

    # Calculate statistics for each period
    period_stats = {
        f"{name}_stats": calculate_statistics(period_data)
        for name, period_data in period_returns.items()
    }

    # Prepare sector ETF data
    sector_returns = prepare_sector_data(returns)

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
