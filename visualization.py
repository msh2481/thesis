import mplfinance as mpf
import numpy as np
import pandas as pd
from agent import Agent, Candle, Trade
from beartype import beartype as typed
from matplotlib import pyplot as plt


@typed
def trades_to_candles(trades: list[Trade], period: float = 1.0) -> list[Candle]:
    """Convert trades to a list of candles with given period length"""
    if not trades:
        return []

    candles: list[Candle] = []
    period_trades: list[Trade] = []
    current_period = int(trades[0].timestamp / period)

    for trade in trades:
        trade_period = int(trade.timestamp / period)

        if trade_period > current_period:
            # Create candle for completed period
            candle = Candle.from_trades(period_trades, float(current_period * period))
            if candle:
                candles.append(candle)

            # Handle any skipped periods
            while current_period < trade_period - 1:
                current_period += 1
                # Create empty candle for skipped period
                last_price = candles[-1].close if candles else trades[0].price
                candles.append(
                    Candle(
                        timestamp=float(current_period * period),
                        open=last_price,
                        high=last_price,
                        low=last_price,
                        close=last_price,
                        volume=0.0,
                        num_trades=0,
                    )
                )

            current_period = trade_period
            period_trades = [trade]
        else:
            period_trades.append(trade)

    # Handle last period
    if period_trades:
        candle = Candle.from_trades(period_trades, float(current_period * period))
        if candle:
            candles.append(candle)

    return candles


@typed
def visualize_trades(
    trades: list[Trade],
    fair_prices: list[float],
    use_candles: bool = True,
    filename: str | None = None,
):
    if use_candles:
        candles = trades_to_candles(trades, period=5.0)
        df = pd.DataFrame(
            [
                {
                    "Date": pd.Timestamp("2024-01-01")
                    + pd.Timedelta(seconds=int(c.timestamp)),
                    "Open": c.open,
                    "High": c.high,
                    "Low": c.low,
                    "Close": c.close,
                    "Volume": c.volume,
                }
                for c in candles
            ]
        )
        df.set_index("Date", inplace=True)

        # For candlestick charts, we need to save within the mpf.plot call
        mpf.plot(
            df,
            type="candle",
            title="Simulated Market Data - Candles",
            ylabel="Price",
            volume=True,
            style="charles",
            figsize=(15, 7),
            savefig=filename,
        )
    else:
        plt.figure(figsize=(15, 7))

        # Plot trades
        trade_times = [
            pd.Timestamp("2024-01-01") + pd.Timedelta(seconds=int(t.timestamp))
            for t in trades
        ]
        trade_prices = [t.price for t in trades]
        trade_sizes = [t.size for t in trades]
        max_size = max(trade_sizes)
        normalized_sizes = [50 * (s / max_size) for s in trade_sizes]

        plt.scatter(
            trade_times,
            trade_prices,
            s=normalized_sizes,
            alpha=0.5,
            c="blue",
            label="Trades",
            zorder=2,
        )
        # Plot fair price line
        fair_times = [
            pd.Timestamp("2024-01-01") + pd.Timedelta(seconds=int(t))
            for t in range(len(fair_prices))
        ]
        plt.plot(fair_times, fair_prices, "r--", lw=2, label="Fair Price")
        plt.title("Individual Trades and Fair Price")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.grid(True)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        if filename:
            plt.savefig(filename)
        else:
            plt.show()


@typed
def visualize_agents(
    agents: list[Agent],
    title: str = "Agent Parameter Distributions",
    filename: str | None = None,
):
    """
    Visualize agent parameters and state
    """
    param_data = {
        "Alpha (Insight)": [a.alpha for a in agents],
        "K (Smoothing)": [a.k for a in agents],
        "Sigma (Noise)": [a.sigma for a in agents],
        "Initial Wealth": [a.initial_wealth for a in agents],
        "Current Wealth": [a.wealth for a in agents],
        "Wealth Change %": [(a.wealth / a.initial_wealth - 1) * 100 for a in agents],
        "Position": [a.position for a in agents],
        "Balance": [a.balance for a in agents],
        "Aggressiveness": [a.aggressiveness for a in agents],
        "Delay": [a.delay for a in agents],
        "Uncertainty": [a.uncertainty for a in agents],
    }

    # Create distribution plots
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(title, fontsize=16)
    axes = axes.flatten()

    for i, (name, values) in enumerate(param_data.items()):
        ax = axes[i]
        ax.hist(values, bins=50, density=True)
        ax.set_title(name)
        ax.grid(True)

        stats = f"Mean: {np.mean(values):.3f}\nStd: {np.std(values):.3f}"
        ax.text(
            0.95,
            0.95,
            stats,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    # Remove unused subplots
    for ax in axes[len(param_data) :]:
        ax.remove()

    plt.tight_layout()

    if filename:
        plt.savefig(filename)
    else:
        plt.show()
