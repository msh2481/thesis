import matplotlib.pyplot as plt
import numpy as np
import torch as t
from beartype import beartype as typed
from beartype.typing import Callable
from envs.diff_stock_trading_env import compute_optimal_1d, DiffStockTradingEnv
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
def detrend(prices: Float[TT, "time_dim stock_dim"]) -> Float[TT, "time_dim stock_dim"]:
    """Detrend prices."""
    initial_prices = prices[0:1]
    final_prices = prices[-1:]
    trend_part = initial_prices * (final_prices / initial_prices).mean() ** (
        t.linspace(0, 1, len(prices))[:, None]
    )
    return prices / trend_part


@typed
def gen_wiener(
    n_steps: int, n_stocks: int, detrend: bool = False
) -> Float[TT, "time_dim stock_dim"]:
    """Generate price array with random walks."""
    price_array = (1 + t.randn((n_steps, n_stocks)) / n_steps**0.5).log()
    price_array = t.cumsum(price_array, dim=0).exp()
    start_price = max(10, 5 - price_array.min())
    price_array += start_price
    if detrend:
        price_array = detrend(price_array)
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


def gen_pair_trading(
    n_steps: int, n_stocks: int, alpha: float = 0.1
) -> Float[TT, "time_dim stock_dim"]:
    """Generate price array with pair trading."""
    # 1 -> n_stocks here to remove mutual information
    common = gen_wiener(n_steps, 1)
    noise = gen_ma(n_steps, n_stocks)
    result = common + alpha * noise
    return result


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
    env = create_env(gen_wiener, 100, 2)
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


def test_detrend():
    price_array = gen_wiener(500, 5, detrend=True)
    plt.plot(price_array)
    plt.show()


def test_pair_trading():
    price_array = gen_pair_trading(300, 30, alpha=0.1)
    plt.plot(price_array, lw=0.5)
    plt.show()


def test_optimal_1d():
    prices = gen_wiener(100, 1)[:, 0]
    value, positions = compute_optimal_1d(prices)
    env = DiffStockTradingEnv(
        prices[:, None],
        t.zeros_like(prices)[:, None, None],
        initial_position=t.zeros((1,)),
    )
    env.reset_state()
    # compute returns using env
    returns = t.zeros_like(prices)
    for i in range(len(prices) - 1):
        returns[i] = env.make_step(positions[i, None])[1]
    cumreturns = t.cumsum(returns, dim=0)
    env_value = env.state.value()
    logger.info(
        f"Optimal value: {value.item():.2f} | Env value: {env_value.item():.2f}"
    )

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(prices, label="Prices")
    plt.grid(True)
    plt.legend()
    plt.title("Prices")

    plt.subplot(3, 1, 2)
    plt.plot(positions, label="Positions")
    plt.grid(True)
    plt.legend()
    plt.title("Positions")

    plt.subplot(3, 1, 3)
    plt.plot(cumreturns, label="Returns")
    plt.grid(True)
    plt.legend()
    plt.title("Returns")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_optimal_1d()
