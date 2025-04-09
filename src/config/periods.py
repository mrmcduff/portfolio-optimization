"""
Period Configuration Module

This module defines the date ranges for different analysis periods used throughout the project.
"""

from typing import Dict, Tuple

# Define period date ranges that will be used across the project
PERIOD_RANGES: Dict[str, Tuple[str, str]] = {
    # The financial crisis period (2008-2013)
    "financial_crisis": ("2008-01-02", "2013-01-02"),
    # Post-crisis period (2012-2018)
    "post_crisis": ("2012-01-01", "2018-12-31"),
    # Recent period (2019-2023)
    "recent": ("2019-01-01", "2023-12-31"),
}


# Function to get period ranges
def get_period_range(period_name: str) -> Tuple[str, str]:
    """
    Get the date range for a specific period.

    Parameters:
    -----------
    period_name : str
        Name of the period to get the range for

    Returns:
    --------
    Tuple[str, str]
        (start_date, end_date) for the specified period
    """
    if period_name not in PERIOD_RANGES:
        raise ValueError(
            f"Unknown period: {period_name}. Available periods: {list(PERIOD_RANGES.keys())}"
        )

    return PERIOD_RANGES[period_name]
