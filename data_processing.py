""" 
Based on https://github.com/AI4Finance-Foundation/FinRL-Meta/blob/master/meta/data_processors/yahoofinance.py
"""

import os
from datetime import datetime

import exchange_calendars as tc
import numpy as np
import pandas as pd
import pytz
import yfinance as yf
from beartype import beartype as typed
from beartype.typing import Callable
from loguru import logger
from matplotlib import pyplot as plt
from talipp.indicators import MACD, RSI
from tqdm.auto import tqdm

# TODO: run LLMs to check this code
# TODO: find small basis for EMA, TEMA, SMA, SMMA, Moving Linear Regression, TRIX, etc.
# TODO: find optimal amount of cash to diversify S&P500
# TODO: plot cumulative volumes per price level


class BaseSource:
    # Add class constants
    REQUIRED_COLUMNS = {"date", "time", "close"}
    FLOAT_PRECISION = 3
    DEFAULT_OUTPUT_FILENAME = "dataset.csv"

    @typed
    def save_data(self, path: str) -> None:
        try:
            path, filename = self._parse_save_path(path)
            os.makedirs(path, exist_ok=True)
            self.df.to_csv(
                path + filename, index=False, float_format=f"%.{self.FLOAT_PRECISION}f"
            )
        except Exception as e:
            logger.exception(f"Failed to save data: {e}")
            raise

    @typed
    def _parse_save_path(self, path: str) -> tuple[str, str]:
        """Extract path and filename from full path."""
        if ".csv" in path:
            path_parts = path.split("/")
            return "/".join(path_parts[:-1] + [""]), path_parts[-1]
        else:
            return (
                path if path.endswith("/") else f"{path}/"
            ), self.DEFAULT_OUTPUT_FILENAME

    def add_technical_indicator(
        self,
        indicators: dict[str, Callable[[pd.DataFrame], pd.DataFrame]],
        drop_na_timesteps: int = 1,
    ):
        """
        calculate technical indicators
        use stockstats/talib package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        self.df.reset_index(drop=True, inplace=True)
        if "level_1" in self.df.columns:
            self.df.drop(columns=["level_1"], inplace=True)
        if "level_0" in self.df.columns and "ticker" not in self.df.columns:
            self.df.rename(columns={"level_0": "ticker"}, inplace=True)

        logger.info(f"Technical indicators: {indicators}")
        unique_ticker = self.df.ticker.unique()
        for indicator, func in indicators.items():
            logger.info(f"Processing indicator: {indicator}")
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                try:
                    input_df = self.df[self.df.ticker == unique_ticker[i]]
                    assert all(
                        column in input_df.columns
                        for column in ["open", "high", "low", "close", "volume"]
                    ), "Input dataframe must contain columns: open, high, low, close, volume"
                    output_df = func(input_df)
                    time_to_drop = output_df[output_df.isna().any(axis=1)][
                        "time"
                    ].unique()
                    if len(time_to_drop) > 30:
                        logger.warning(
                            f"Time to drop for {indicator} for {unique_ticker[i]}: {time_to_drop}"
                        )
                        logger.warning(
                            self.df[self.df.ticker == unique_ticker[i]].head(10)
                        )
                    indicator_df = pd.concat(
                        [indicator_df, output_df],
                        axis=0,
                        join="outer",
                        ignore_index=True,
                    )
                except Exception as e:
                    logger.exception(e)
            if not indicator_df.empty:
                self.df = self.df.merge(
                    indicator_df,
                    on=["ticker", "time"],
                    how="left",
                )

        if drop_na_timesteps:
            old_df = self.df.copy()
            time_to_drop = self.df[self.df.isna().any(axis=1)].time.unique()
            logger.info(f"Times to drop: {time_to_drop}")
            self.df = self.df[~self.df.time.isin(time_to_drop)]
        self.df.reset_index(drop=True, inplace=True)
        logger.info("Succesfully add technical indicators")

    def df_to_array(self, tech_indicator_list: list[str]):
        unique_ticker = self.df.ticker.unique()
        logger.warning(f"Columns: {self.df.columns.values.tolist()}")
        price_arrays = [
            self.df[self.df.ticker == ticker].close for ticker in unique_ticker
        ]
        price_array = np.column_stack(price_arrays)
        not_tech_indicator_list = ["ticker", "time", "time_idx"]
        full_list = [
            i
            for i in self.df.columns.values.tolist()
            if i not in not_tech_indicator_list
        ]
        logger.info(f"Indicators available: {full_list}")
        tech_df = self.df[tech_indicator_list]
        logger.debug(tech_df.head(10))
        tech_array = np.hstack(
            [
                self.df.loc[(self.df.ticker == ticker), tech_indicator_list]
                for ticker in unique_ticker
            ]
        )
        logger.info("Successfully transformed into array")
        return price_array, tech_array

    # Standard_time_interval  s: second, m: minute, h: hour, d: day, w: week, M: month, q: quarter, y: year
    # Output time_interval of the processor
    def calc_nonstandard_time_interval(self) -> str:
        assert self.data_source == "yahoofinance"
        # Nonstandard time interval: ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d","1wk", "1mo", "3mo"]
        time_intervals = [
            "1m",
            "2m",
            "5m",
            "15m",
            "30m",
            "60m",
            "90m",
            "1h",
            "1d",
            "5d",
            "1w",
            "1M",
            "3M",
        ]
        assert self.time_interval in time_intervals, (
            "This time interval is not supported. Supported time intervals: "
            + ",".join(time_intervals)
        )
        if "w" in self.time_interval:
            return self.time_interval + "k"
        elif "M" in self.time_interval:
            return self.time_interval[:-1] + "mo"
        else:
            return self.time_interval

    def load_data(self, path):
        assert ".csv" in path
        self.df = pd.read_csv(path)
        columns = self.df.columns
        logger.info(f"{path} loaded")
        # Check loaded file
        assert "date" in columns or "time" in columns, "date or time column not found"
        assert "close" in columns, "close column not found"


class YahooFinance(BaseSource):
    SUPPORTED_INTERVALS = {
        "1m",
        "2m",
        "5m",
        "15m",
        "30m",
        "60m",
        "90m",
        "1h",
        "1D",
        "5D",
        "1w",
        "1M",
        "3M",
    }
    TRADING_HOURS = {"start": "09:30:00", "end": "16:00:00"}

    @typed
    def __init__(
        self,
        data_source: str,
        start_date: str | datetime,
        end_date: str | datetime,
        time_interval: str,
        **kwargs,
    ):
        if time_interval not in self.SUPPORTED_INTERVALS:
            raise ValueError(
                f"Unsupported time interval. Must be one of: {self.SUPPORTED_INTERVALS}"
            )

        self.data_source = data_source
        self.start_date = (
            start_date
            if isinstance(start_date, str)
            else start_date.strftime("%Y-%m-%d")
        )
        self.end_date = (
            end_date if isinstance(end_date, str) else end_date.strftime("%Y-%m-%d")
        )
        self.time_interval = time_interval

    @typed
    def download_data(self, ticker_list: list[str], save_path: str) -> None:
        """
        Download data for given tickers and save to CSV.

        Args:
            ticker_list: List of stock tickers to download
            save_path: Path to save the resulting CSV file
        """
        try:
            self._download_ticker_data(ticker_list)
            self.save_data(save_path)
            logger.info(
                f"Download complete! Dataset saved to {save_path}.\n"
                f"Shape of DataFrame: {self.df.shape}"
            )
        except Exception as e:
            logger.error(f"Failed to download data: {e}")
            raise

    @typed
    def _download_ticker_data(self, ticker_list: list[str]) -> None:
        """Download raw data for each ticker."""
        self.time_zone = pytz.utc
        self.df = pd.DataFrame()
        for ticker in ticker_list:
            temp_df = yf.download(
                ticker,
                start=self.start_date,
                end=self.end_date,
                interval=self.time_interval,
            )
            temp_df["ticker"] = ticker
            self.df = pd.concat([self.df, temp_df], axis=0, join="outer")
        self.df.reset_index(inplace=True)
        self.df.columns = [
            "date",
            "open",
            "high",
            "low",
            "close",
            "adjusted_close",
            "volume",
            "ticker",
        ]
        self.df["day"] = self.df["date"].dt.dayofweek
        logger.debug(self.df)
        self.df["date"] = self.df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
        self.df.dropna(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.df.sort_values(by=["date", "ticker"], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

    @typed
    def _get_times(self) -> list[pd.Timestamp] | list[str]:
        trading_days = self.get_trading_days(start=self.start_date, end=self.end_date)
        if self.time_interval == "1D":
            return trading_days
        elif self.time_interval == "1h":
            times = []
            for day in trading_days:
                current_time = pd.Timestamp(day + " 09:30:00").tz_localize(
                    self.time_zone
                )
                for _ in range(6):
                    times.append(current_time)
                    current_time += pd.Timedelta(hours=1)
            return times
        elif self.time_interval == "1m":
            times = []
            for day in trading_days:
                current_time = pd.Timestamp(day + " 09:30:00").tz_localize(
                    self.time_zone
                )
                for _ in range(390):
                    times.append(current_time)
                    current_time += pd.Timedelta(minutes=1)
            return times
        else:
            raise ValueError(
                "Data clean at given time interval is not supported for YahooFinance data."
            )

    @typed
    def _clean_data_for_ticker(
        self,
        df: pd.DataFrame,
        ticker: str,
        times: list[pd.Timestamp] | list[str],
    ) -> pd.DataFrame | None:
        # logger.debug(f"Clean data for {ticker}")
        columns = ["time", "open", "high", "low", "close", "volume", "day"]
        tmp_df = (
            df[df.ticker == ticker]
            .drop(columns=["close"])
            .rename(columns={"adjusted_close": "close"})
            .set_index("time")
            .reindex(times)
            .reset_index()
        )

        if pd.isna(tmp_df.iloc[0]["close"]):
            logger.warning(f"NaN data on start date for {ticker}. Dropping...")
            return None
            # logger.info("NaN data on start date, fill using first valid data.")
            # first_valid = tmp_df.loc[tmp_df["close"].first_valid_index()]
            # tmp_df.iloc[0][["open", "high", "low", "close", "volume"]] = [
            #     first_valid["close"],
            #     first_valid["close"],
            #     first_valid["close"],
            #     first_valid["close"],
            #     0.0,
            # ]
        tmp_df.loc[pd.isna(tmp_df["volume"]), "volume"] = 0.0
        tmp_df["time_idx"] = range(len(tmp_df))

        # Day := day - time_idx to meaningfully ffill such arithmetic progressions
        tmp_df["day"] = (tmp_df["day"] - tmp_df["time_idx"]) % 7
        tmp_df = tmp_df.infer_objects(copy=False).ffill()
        # Convert back to original day
        tmp_df["day"] = (tmp_df["day"] + tmp_df["time_idx"]) % 7

        tmp_df["ticker"] = ticker
        tmp_df["volume"] = tmp_df["volume"].astype(float)
        tmp_df = tmp_df[["time_idx"] + columns + ["ticker"]]

        return tmp_df

    @typed
    def clean_data(self) -> None:
        df = self.df.copy()
        df = df.rename(columns={"date": "time"})
        ticker_list = np.unique(df.ticker.values)
        new_df = pd.DataFrame()

        times = self._get_times()
        for ticker in tqdm(ticker_list):
            tmp_df = self._clean_data_for_ticker(df, ticker, times)
            if tmp_df is not None:
                new_df = pd.concat([new_df, tmp_df], ignore_index=True)

        logger.info("Data clean all finished!")
        self.df = new_df

    @typed
    def get_trading_days(self, start: str, end: str) -> list[str]:
        nyse = tc.get_calendar("NYSE")
        df = nyse.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
        return [str(day)[:10] for day in df]


def test():
    processor = YahooFinance(
        data_source="YahooFinance",
        start_date="2023-01-01",
        end_date="2024-01-01",
        time_interval="1D",
    )
    sp500_table = pd.read_html(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    )
    sp500_tickers = sp500_table[0]["Symbol"].tolist()
    print(sp500_tickers)
    processor.download_data(sp500_tickers, "data/sp500_2023_2023.csv")
    processor.clean_data()


def rsi_14(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df.close.values
    df["rsi_14"] = RSI(
        period=14,
        input_values=close,
    )
    return df[["time", "ticker", "rsi_14"]]


def macd(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df.close.values
    output = MACD(
        fast_period=12,
        slow_period=26,
        signal_period=9,
        input_values=close,
    )
    df["macd"] = [value.histogram if value else None for value in output]
    return df[["time", "ticker", "macd"]]


def test_2():
    processor = YahooFinance(
        data_source="YahooFinance",
        start_date="2023-01-01",
        end_date="2024-01-01",
        time_interval="1D",
    )
    # processor.load_data("data/test.csv")
    # processor.clean_data()
    # processor.save_data("data/test_clean.csv")

    # processor.load_data("data/sp500_2023_2023.csv")
    # print(len(processor.dataframe.ticker.unique()))
    # processor.clean_data()
    # print(len(processor.dataframe.ticker.unique()))
    # processor.save_data("data/sp500_2023_2023_clean.csv")
    # exit()

    processor.load_data("data/test_clean.csv")
    indicators = {"RSI": rsi_14, "MACD": macd}
    processor.add_technical_indicator(indicators)
    processor.save_data("data/test_clean_tech.csv")
    stocks, tech = processor.df_to_array(["rsi_14", "macd"])
    print(stocks.shape, tech.shape)

    tickers = processor.df.ticker.unique().tolist()
    idx = tickers.index("AAPL")
    stock = stocks[:, idx]
    tech = tech[:, idx :: len(tickers)]
    print(stock.shape, tech.shape)

    stock = (stock - stock.mean()) / stock.std()
    tech = (tech - tech.mean(axis=0)) / tech.std(axis=0)
    plt.plot(stock, label="stock")
    plt.plot(tech[:, 0], label="rsi")
    plt.plot(tech[:, 1], label="macd")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_2()