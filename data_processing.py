""" 
Based on https://github.com/AI4Finance-Foundation/FinRL-Meta/blob/master/meta/data_processors/yahoofinance.py
"""

import os

import exchange_calendars as tc
import numpy as np
import pandas as pd
import pytz
import stockstats
import yfinance as yf
from beartype import beartype as typed
from loguru import logger

# TODO: run LLMs to check this code


class BaseSource:
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
                    temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator["ticker"] = unique_ticker[i]
                    temp_indicator["time"] = self.dataframe[
                        self.dataframe.tic == unique_ticker[i]
                    ]["time"].to_list()
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
            time_to_drop = self.dataframe[
                self.dataframe.isna().any(axis=1)
            ].time.unique()
            self.dataframe = self.dataframe[~self.dataframe.time.isin(time_to_drop)]
        logger.info("Succesfully add technical indicators")

    def df_to_array(self, tech_indicator_list: list[str]):
        unique_ticker = self.dataframe.ticker.unique()
        price_array = np.column_stack(
            [
                self.dataframe[self.dataframe.ticker == tic].close
                for tic in unique_ticker
            ]
        )
        common_tech_indicator_list = [
            i
            for i in tech_indicator_list
            if i in self.dataframe.columns.values.tolist()
        ]
        tech_array = np.hstack(
            [
                self.dataframe.loc[
                    (self.dataframe.ticker == tic), common_tech_indicator_list
                ]
                for tic in unique_ticker
            ]
        )
        logger.info("Successfully transformed into array")
        return price_array, tech_array

    # standard_time_interval  s: second, m: minute, h: hour, d: day, w: week, M: month, q: quarter, y: year
    # output time_interval of the processor
    def calc_nonstandard_time_interval(self) -> str:
        assert self.data_source == "yahoofinance"
        # nonstandard_time_interval: ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d","1wk", "1mo", "3mo"]
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

    def save_data(self, path):
        if ".csv" in path:
            path = path.split("/")
            filename = path[-1]
            path = "/".join(path[:-1] + [""])
        else:
            if path[-1] == "/":
                filename = "dataset.csv"
            else:
                filename = "/dataset.csv"

        os.makedirs(path, exist_ok=True)
        self.dataframe.to_csv(path + filename, index=False)

    def load_data(self, path):
        assert ".csv" in path
        self.dataframe = pd.read_csv(path)
        columns = self.dataframe.columns
        logger.info(f"{path} loaded")
        # check loaded file
        assert "date" in columns or "time" in columns, "date or time column not found"
        assert "close" in columns, "close column not found"


class YahooFinance(BaseSource):
    @typed
    def __init__(
        self,
        data_source: str,
        start_date: str,
        end_date: str,
        time_interval: str,
        **kwargs,
    ):
        self.data_source = data_source
        self.start_date = start_date
        self.end_date = end_date
        self.time_interval = time_interval

    @typed
    def download_data(self, ticker_list: list[str], save_path: str) -> None:
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
        self.save_data(save_path)
        logger.info(
            f"Download complete! Dataset saved to {save_path}.\n"
            f"Shape of DataFrame: {self.dataframe.shape}"
        )

    @typed
    def _get_times(self) -> list[pd.Timestamp] | list[str]:
        trading_days = self.get_trading_days(start=self.start_date, end=self.end_date)
        if self.time_interval == "1D":
            return trading_days
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
    ) -> pd.DataFrame:
        logger.info(f"Clean data for {ticker}")
        tmp_df = pd.DataFrame(
            columns=[
                "open",
                "high",
                "low",
                "close",
                "adjusted_close",
                "volume",
            ],
            index=times,
        )
        # Get data for current ticker
        ticker_df = df[df.ticker == ticker]
        # Fill empty DataFrame using original data
        for i in range(ticker_df.shape[0]):
            tmp_df.loc[ticker_df.iloc[i]["time"]] = ticker_df.iloc[i][
                [
                    "open",
                    "high",
                    "low",
                    "close",
                    "adjusted_close",
                    "volume",
                ]
            ]

        # If close on start date is NaN, fill data with first valid close and set volume to 0
        if str(tmp_df.iloc[0]["close"]) == "nan":
            logger.info("NaN data on start date, fill using first valid data.")
            for i in range(tmp_df.shape[0]):
                if str(tmp_df.iloc[i]["close"]) != "nan":
                    first_valid_close = tmp_df.iloc[i]["close"]
                    first_valid_adjclose = tmp_df.iloc[i]["adjusted_close"]

            tmp_df.iloc[0] = [
                first_valid_close,
                first_valid_close,
                first_valid_close,
                first_valid_close,
                first_valid_adjclose,
                0.0,
            ]

        # Fill NaN data with previous close and set volume to 0.
        for i in range(tmp_df.shape[0]):
            if str(tmp_df.iloc[i]["close"]) == "nan":
                previous_close = tmp_df.iloc[i - 1]["close"]
                previous_adjusted_close = tmp_df.iloc[i - 1]["adjusted_close"]
                if str(previous_close) == "nan":
                    raise ValueError
                tmp_df.iloc[i] = [
                    previous_close,
                    previous_close,
                    previous_close,
                    previous_close,
                    previous_adjusted_close,
                    0.0,
                ]

        # Merge single ticker data to new DataFrame
        tmp_df = tmp_df.astype(float)
        tmp_df["ticker"] = ticker
        logger.info(f"Data clean for {ticker} is finished.")
        return tmp_df

    @typed
    def clean_data(self) -> None:
        df = self.dataframe.copy()
        df = df.rename(columns={"date": "time"})
        ticker_list = np.unique(df.ticker.values)
        new_df = pd.DataFrame()

        times = self._get_times()
        for ticker in ticker_list:
            tmp_df = self._clean_data_for_ticker(df, ticker, times)
            new_df = pd.concat([new_df, tmp_df], ignore_index=True)

        # reset index and rename columns
        new_df = new_df.reset_index()
        new_df = new_df.rename(columns={"index": "time"})
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
        start_date="2024-01-01",
        end_date="2024-01-05",
        time_interval="1D",
    )
    processor.download_data(["AAPL", "GOOG"], "data/test.csv")
    processor.clean_data()


if __name__ == "__main__":
    test()
