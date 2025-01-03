import matplotlib.pyplot as plt
import numpy as np
import torch as t
from beartype import beartype as typed
from beartype.typing import Callable
from envs.diff_stock_trading_env import DiffStockTradingEnv
from jaxtyping import Float
from loguru import logger
from torch import Tensor as TT


@typed
def make_ema_tech(
    price_array: Float[TT, "time_dim stock_dim"],
    periods: list[int] = [1, 4, 16],
) -> Float[TT, "time_dim stock_dim tech_dim"]:
    """Generate technical indicators (EMAs) for each stock."""
    time_dim, stock_dim = price_array.shape
    tech_dim = len(periods)
    tech_array = t.zeros((time_dim, stock_dim, tech_dim))

    for i, period in enumerate(periods):
        alpha = 1.0 / (period + 1)
        ema = price_array.clone()
        for j in range(1, time_dim):
            ema[j] = alpha * price_array[j] + (1 - alpha) * ema[j - 1]
        tech_array[:, :, i] = ema

    return tech_array


@typed
def generate_random_walks(
    n_steps: int, n_stocks: int
) -> Float[TT, "time_dim stock_dim"]:
    """Generate price array with random walks."""
    price_array = t.randn((n_steps, n_stocks)) / n_steps**0.5
    price_array = t.cumsum(price_array, dim=0)
    price_array = t.exp(price_array)
    return price_array


@typed
def gen_ma(n_steps: int, n_stocks: int, dt: int = 5) -> Float[TT, "time_dim stock_dim"]:
    """Generate price array following MA process."""
    noise = t.randn((n_steps + dt, n_stocks))
    price_array = t.zeros((n_steps, n_stocks))
    for i in range(n_steps):
        price_array[i] = noise[i : i + dt].mean(dim=0).exp()
    return price_array


@typed
def gen_trend(n_steps: int, n_stocks: int) -> Float[TT, "time_dim stock_dim"]:
    """Generate price array with trends."""
    price_array = t.zeros((n_steps, n_stocks))
    upward = t.rand((n_stocks,)) > 0.5
    new_price = t.ones((n_stocks,))

    for i in range(n_steps):
        deltas = t.randn(n_stocks).exp() / 100
        sign = (2 * upward - 1).float()
        new_price *= (sign * deltas).exp()
        flip_prob = t.sigmoid(new_price.log() * sign - 3)
        flip = t.rand((n_stocks,)) < flip_prob
        upward = t.where(flip, ~upward, upward)
        price_array[i] = new_price

    return price_array


@typed
def create_env(
    price_generator: Callable, n_steps: int, n_stocks: int, **kwargs
) -> DiffStockTradingEnv:
    """Create environment with given price generator."""
    price_array = price_generator(n_steps, n_stocks, **kwargs)
    tech_array = make_ema_tech(price_array)
    return DiffStockTradingEnv(price_array, tech_array)


def assert_good_tensor(tensor: Float[TT, "..."]):
    assert not t.isnan(tensor).any()
    assert not t.isinf(tensor).any()
    assert t.isfinite(tensor).all()


def test_corner_cases():
    env = create_env(generate_random_walks, 100, 2)
    state, _ = env.reset_state()
    assert_good_tensor(state.features())

    for _ in range(10**5):
        # action = t.randn(env.stock_dim).exp()
        # action /= action.sum() + t.randn(()).exp()
        action = t.ones(env.stock_dim) / (env.stock_dim + 1)
        state, reward, terminated, truncated, _ = env.make_step(action)
        logger.debug(
            f"Reward: {reward.item():.2f} | Value: {state.value().item():.2f} | Position: {state.position().numpy()}"
        )
        done = terminated or truncated
        assert_good_tensor(state.features())
        assert_good_tensor(reward)
        if done:
            break


@typed
def plot_returns(
    returns: list[float],
    returns_ema: list[float],
    return_stds: list[float],
    gradients: list[float],
    val_returns: list[float] | None = None,
    val_returns_ema: list[float] | None = None,
    val_period: int = 5,
    eval_interval: int = 1,
    avg_returns: list[float] | None = None,
    avg_returns_ema: list[float] | None = None,
    val_avg_returns: list[float] | None = None,
    val_avg_returns_ema: list[float] | None = None,
) -> None:
    """Plot training and validation returns."""
    l = t.tensor(returns) - t.tensor(return_stds)
    r = t.tensor(returns) + t.tensor(return_stds)
    plt.close()
    plt.clf()

    n_plots = 3 if val_returns else 2

    plt.figure(figsize=(10, 10))
    # Training returns plot
    plt.subplot(n_plots, 1, 1)
    plt.title("Training Returns")
    plt.axhline(0, color="k", linestyle="--")
    plt.grid(True, which="major", alpha=0.5)
    plt.grid(True, which="minor", alpha=0.1)
    plt.minorticks_on()
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    plt.fill_between(range(len(returns)), l, r, alpha=0.2, color="red")
    plt.plot(returns, "r-", lw=0.5, label="Current Policy")
    plt.plot(returns_ema, "r--", label="Current EMA")
    if avg_returns is not None:
        eval_x = list(range(0, len(returns), eval_interval))
        plt.plot(eval_x, avg_returns, "b-", lw=0.5, label="Polyak Average")
        plt.plot(eval_x, avg_returns_ema, "b--", label="Polyak EMA")
    plt.legend()
    returns_95 = t.quantile(t.tensor(returns), 0.95)
    returns_5 = t.quantile(t.tensor(returns), 0.05)
    plt.ylim(returns_5 - 0.1, returns_95 + 0.1)

    # Gradient plot
    plt.subplot(n_plots, 1, 2)
    plt.title("Gradient Norms")
    plt.plot(gradients)
    plt.yscale("log")
    plt.grid(True, which="major", alpha=0.3)

    # Validation returns plot if applicable
    if val_returns:
        plt.subplot(n_plots, 1, 3)
        plt.title("Validation Returns")
        plt.grid(True, which="major", alpha=0.5)
        plt.grid(True, which="minor", alpha=0.1)
        plt.minorticks_on()
        plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.1))
        val_x = list(range(0, len(returns), val_period))
        plt.plot(val_x, val_returns, "r-", lw=0.5, label="Current Policy")
        plt.plot(val_x, val_returns_ema, "r--", label="Current EMA")
        if val_avg_returns is not None:
            plt.plot(val_x, val_avg_returns, "b-", lw=0.5, label="Polyak Average")
            plt.plot(val_x, val_avg_returns_ema, "b--", label="Polyak EMA")
        plt.legend()
        if val_returns:
            val_returns_95 = t.quantile(t.tensor(val_returns), 0.95)
            val_returns_5 = t.quantile(t.tensor(val_returns), 0.05)
            plt.ylim(val_returns_5 - 0.1, val_returns_95 + 0.1)
        plt.axhline(0, color="k", linestyle="--")

    plt.tight_layout()
    plt.savefig("logs/info.png")
    plt.close()


if __name__ == "__main__":
    test_corner_cases()
