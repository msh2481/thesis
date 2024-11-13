""" 
Based on https://github.com/AI4Finance-Foundation/FinRL-Meta/blob/master/meta/data_processors/yahoofinance.py
"""

import os
from datetime import datetime

import exchange_calendars as tc
import numpy as np
import pandas as pd
import pytz
import stockstats
import yfinance as yf
from beartype import beartype as typed
from loguru import logger
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

# TODO: run LLMs to check this code
# TODO: find small basis for EMA, TEMA, SMA, SMMA, Moving Linear Regression, TRIX, etc.
# TODO: find optimal amount of cash to diversify S&P500


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
            self.dataframe.to_csv(
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
        tech_indicator_list: list[str],
        drop_na_timesteps: int = 1,
    ):
        """
        calculate technical indicators
        use stockstats/talib package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        if "date" in self.dataframe.columns.values.tolist():
            self.dataframe.rename(columns={"date": "time"}, inplace=True)

        if self.data_source == "ccxt":
            self.dataframe.rename(columns={"index": "time"}, inplace=True)

        self.dataframe.reset_index(drop=False, inplace=True)
        if "level_1" in self.dataframe.columns:
            self.dataframe.drop(columns=["level_1"], inplace=True)
        if (
            "level_0" in self.dataframe.columns
            and "ticker" not in self.dataframe.columns
        ):
            self.dataframe.rename(columns={"level_0": "ticker"}, inplace=True)

        logger.info(f"tech_indicator_list: {tech_indicator_list}")
        stock = stockstats.StockDataFrame.retype(self.dataframe)
        unique_ticker = stock.ticker.unique()
        for indicator in tech_indicator_list:
            logger.info(f"indicator: {indicator}")
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                try:
                    temp_indicator = stock[stock.ticker == unique_ticker[i]][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator["ticker"] = unique_ticker[i]
                    temp_indicator["time"] = self.dataframe[
                        self.dataframe.ticker == unique_ticker[i]
                    ]["time"].to_list()
                    time_to_drop = temp_indicator[temp_indicator.isna().any(axis=1)][
                        "time"
                    ].unique()
                    if len(time_to_drop) > 2:
                        logger.warning(
                            f"Time to drop for {indicator} for {unique_ticker[i]}: {time_to_drop}"
                        )
                        print(stock[stock.ticker == unique_ticker[i]].head(10))
                    indicator_df = pd.concat(
                        [indicator_df, temp_indicator],
                        axis=0,
                        join="outer",
                        ignore_index=True,
                    )
                except Exception as e:
                    logger.exception(e)
            if not indicator_df.empty:
                self.dataframe = self.dataframe.merge(
                    indicator_df[["ticker", "time", indicator]],
                    on=["ticker", "time"],
                    how="left",
                )

        self.dataframe.sort_values(by=["time", "ticker"], inplace=True)
        if drop_na_timesteps:
            old_df = self.dataframe.copy()
            time_to_drop = self.dataframe[
                self.dataframe.isna().any(axis=1)
            ].time.unique()
            logger.info(f"Times to drop: {time_to_drop}")
            self.dataframe = self.dataframe[~self.dataframe.time.isin(time_to_drop)]
        logger.info("Succesfully add technical indicators")

    def df_to_array(self, tech_indicator_list: list[str]):
        unique_ticker = self.dataframe.ticker.unique()
        price_arrays = [
            self.dataframe[self.dataframe.ticker == ticker].close
            for ticker in unique_ticker
        ]
        price_array = np.column_stack(price_arrays)
        common_tech_indicator_list = [
            i
            for i in tech_indicator_list
            if i in self.dataframe.columns.values.tolist()
        ]
        tech_array = np.hstack(
            [
                self.dataframe.loc[
                    (self.dataframe.ticker == ticker), common_tech_indicator_list
                ]
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
        self.dataframe = pd.read_csv(path)
        columns = self.dataframe.columns
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
                f"Shape of DataFrame: {self.dataframe.shape}"
            )
        except Exception as e:
            logger.error(f"Failed to download data: {e}")
            raise

    @typed
    def _download_ticker_data(self, ticker_list: list[str]) -> None:
        """Download raw data for each ticker."""
        self.time_zone = pytz.utc
        self.dataframe = pd.DataFrame()
        for ticker in ticker_list:
            temp_df = yf.download(
                ticker,
                start=self.start_date,
                end=self.end_date,
                interval=self.time_interval,
            )
            temp_df["ticker"] = ticker
            self.dataframe = pd.concat([self.dataframe, temp_df], axis=0, join="outer")
        self.dataframe.reset_index(inplace=True)
        self.dataframe.columns = [
            "date",
            "open",
            "high",
            "low",
            "close",
            "adjusted_close",
            "volume",
            "ticker",
        ]
        self.dataframe["day"] = self.dataframe["date"].dt.dayofweek
        logger.debug(self.dataframe)
        self.dataframe["date"] = self.dataframe.date.apply(
            lambda x: x.strftime("%Y-%m-%d")
        )
        self.dataframe.dropna(inplace=True)
        self.dataframe.reset_index(drop=True, inplace=True)
        self.dataframe.sort_values(by=["date", "ticker"], inplace=True)
        self.dataframe.reset_index(drop=True, inplace=True)

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
        df = self.dataframe.copy()
        df = df.rename(columns={"date": "time"})
        ticker_list = np.unique(df.ticker.values)
        new_df = pd.DataFrame()

        times = self._get_times()
        for ticker in tqdm(ticker_list):
            tmp_df = self._clean_data_for_ticker(df, ticker, times)
            if tmp_df is not None:
                new_df = pd.concat([new_df, tmp_df], ignore_index=True)

        logger.info("Data clean all finished!")
        self.dataframe = new_df

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

    processor.load_data("data/sp500_2023_2023_clean.csv")
    indicators = ["rsi", "macd"]
    stocks, tech = processor.df_to_array([])
    processor.add_technical_indicator(indicators)
    stocks, tech = processor.df_to_array(indicators)
    print(stocks.shape, tech.shape)

    tickers = processor.dataframe.ticker.unique().tolist()
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
