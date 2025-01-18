import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from beartype import beartype
from beartype.typing import List, Optional, Union
from loguru import logger


@beartype
def get_price_data(
    tickers: List[str],
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Get historical price data for given tickers.

    Args:
        tickers: List of stock symbols
        start_date: Start date for data collection
        end_date: End date for data collection
        interval: Data frequency ('1d', '1h', etc)

    Returns:
        DataFrame with dates as index and tickers as columns
    """
    # Convert dates to strings if needed
    start_date = (
        start_date if isinstance(start_date, str) else start_date.strftime("%Y-%m-%d")
    )
    end_date = end_date if isinstance(end_date, str) else end_date.strftime("%Y-%m-%d")

    # Sort tickers for consistent column ordering
    tickers = sorted(tickers)

    # Download data for all tickers
    df_list = []
    for ticker in tickers:
        try:
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False,
            )
            if len(data) == 0:
                logger.warning(f"No data available for {ticker}")
                continue
            # Calculate average price
            price = (data["Open"] + data["Close"] + data["High"] + data["Low"]) / 4
            df_list.append(price)
        except Exception as e:
            logger.warning(f"Failed to download {ticker}: {e}")

    # Combine all tickers into one dataframe
    if not df_list:
        raise ValueError("No data was downloaded for any ticker")

    df = pd.concat(df_list, axis=1)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()  # Sort by date
    return df


@beartype
def save_stock_data(
    tickers: List[str], splits: List[tuple[str, str, str]], base_path: str = "data"
) -> None:
    """
    Download and save stock data for different time periods.

    Args:
        tickers: List of stock symbols
        splits: List of (start_date, end_date, suffix) tuples
        base_path: Directory to save the files
    """
    os.makedirs(base_path, exist_ok=True)

    for start_date, end_date, suffix in splits:
        logger.info(f"Processing period {start_date} to {end_date}")
        df = get_price_data(tickers, start_date, end_date)

        # Save as CSV with dates as index
        filename = f"{base_path}/stocks_{suffix}"
        df.to_csv(f"{filename}.csv", date_format="%Y-%m-%d")
        logger.info(f"Saved data to {filename}.csv")


@beartype
def load_stock_data(filename: str) -> np.ndarray:
    """
    Load stock data from CSV and return numpy array of values.

    Args:
        filename: Path to CSV file

    Returns:
        Numpy array of shape (n_timesteps, n_stocks)
    """
    df = pd.read_csv(filename, index_col=0, parse_dates=True)
    return df.values


@beartype
def plot_stocks(
    df: pd.DataFrame,
    tickers: Optional[List[str]] = None,
    normalize: bool = True,
) -> None:
    """
    Plot stock prices over time.

    Args:
        df: DataFrame with dates as index and tickers as columns
        tickers: List of tickers to plot (plots all if None)
        normalize: Whether to normalize prices to starting value
        window: Rolling average window (no smoothing if None)
        figsize: Figure size (width, height)
    """

    data = df.copy()
    if tickers:
        data = data[sorted(tickers)]

    if normalize:
        data = data / data.iloc[0]

    print(data)

    data.plot()
    plt.title("Stock Prices Over Time")
    plt.xlabel("Date")
    plt.ylabel("Normalized Price" if normalize else "Price")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def example_usage():
    # Example usage of the functions
    tickers = ["AAPL", "MSFT", "GOOGL"]
    splits = [
        ("2009-01-01", "2020-01-01", "old"),
        ("2020-01-01", "2022-01-01", "new"),
        ("2022-01-01", "2024-01-01", "test"),
    ]

    # Save data for different periods
    save_stock_data(tickers, splits)

    # Load and plot recent data
    df = get_price_data(tickers, "2023-01-01", "2024-01-01")
    plot_stocks(df, normalize=True)

    # Load data as numpy array
    data = load_stock_data("data/stocks_test.csv")
    print(f"Loaded data shape: {data.shape}")


if __name__ == "__main__":
    example_usage()
