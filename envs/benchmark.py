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


# -----------------------------------------------------------------------------
# Operators used in alphas
# -----------------------------------------------------------------------------


@typed
def ts_max(x: Float[TT, "time stocks"], d: int) -> Float[TT, "time stocks"]:
    """Time-series max over the past d days."""
    result = t.zeros_like(x)
    for i in range(len(x)):
        start_idx = max(0, i - d + 1)
        result[i] = t.max(x[start_idx : i + 1], dim=0)[0]
    return result


@typed
def ts_min(x: Float[TT, "time stocks"], d: int) -> Float[TT, "time stocks"]:
    """Time-series min over the past d days."""
    result = t.zeros_like(x)
    for i in range(len(x)):
        start_idx = max(0, i - d + 1)
        result[i] = t.min(x[start_idx : i + 1], dim=0)[0]
    return result


@typed
def ts_argmax(x: Float[TT, "time stocks"], d: int) -> Float[TT, "time stocks"]:
    """Returns days since the maximum in the past d days occurred."""
    result = t.zeros_like(x)
    for i in range(len(x)):
        start_idx = max(0, i - d + 1)
        window = x[start_idx : i + 1]
        max_idx = t.argmax(window, dim=0)
        result[i] = (len(window) - 1 - max_idx).float()
    return result


@typed
def ts_argmin(x: Float[TT, "time stocks"], d: int) -> Float[TT, "time stocks"]:
    """Returns days since the minimum in the past d days occurred."""
    result = t.zeros_like(x)
    for i in range(len(x)):
        start_idx = max(0, i - d + 1)
        window = x[start_idx : i + 1]
        min_idx = t.argmin(window, dim=0)
        result[i] = (len(window) - 1 - min_idx).float()
    return result


@typed
def ts_rank(x: Float[TT, "time stocks"], d: int) -> Float[TT, "time stocks"]:
    """Time-series rank in the past d days."""
    result = t.zeros_like(x)
    for i in range(len(x)):
        start_idx = max(0, i - d + 1)
        window = x[start_idx : i + 1]
        ranks = t.argsort(t.argsort(window, dim=0), dim=0).float()
        result[i] = ranks[-1] / max(1, len(window) - 1)
    return result


@typed
def rank(x: Float[TT, "time stocks"]) -> Float[TT, "time stocks"]:
    """Cross-sectional rank."""
    return t.argsort(t.argsort(x, dim=1), dim=1).float() / (x.shape[1] - 1)


@typed
def delay(x: Float[TT, "time stocks"], d: int) -> Float[TT, "time stocks"]:
    """Value of x d days ago."""
    result = t.zeros_like(x)
    result[d:] = x[:-d]
    return result


@typed
def correlation(
    x: Float[TT, "time stocks"], y: Float[TT, "time stocks"], d: int
) -> Float[TT, "time stocks"]:
    """Time-serial correlation of x and y for the past d days."""
    result = t.zeros_like(x)
    for i in range(1, len(x)):
        start_idx = max(0, i - d + 1)
        x_window = x[start_idx : i + 1].clone()
        y_window = y[start_idx : i + 1].clone()
        x_window -= t.mean(x_window, dim=0)
        y_window -= t.mean(y_window, dim=0)
        x_window /= t.std(x_window, dim=0)
        y_window /= t.std(y_window, dim=0)
        result[i] = t.mean(x_window * y_window, dim=0)
    return result


@typed
def covariance(
    x: Float[TT, "time stocks"], y: Float[TT, "time stocks"], d: int
) -> Float[TT, "time stocks"]:
    """Time-serial covariance of x and y for the past d days."""
    result = t.zeros_like(x)
    for i in range(1, len(x)):
        start_idx = max(0, i - d + 1)
        x_window = x[start_idx : i + 1].clone()
        y_window = y[start_idx : i + 1].clone()
        x_window -= t.mean(x_window, dim=0)
        y_window -= t.mean(y_window, dim=0)
        result[i] = t.mean(x_window * y_window, dim=0)
    return result


@typed
def scale(x: Float[TT, "time stocks"], a: float = 1.0) -> Float[TT, "time stocks"]:
    """Rescale x such that sum(abs(x)) = a."""
    abs_sum = t.sum(t.abs(x), dim=1, keepdim=True)
    return a * x / (abs_sum + 1e-12)  # Add epsilon to prevent division by zero


@typed
def delta(x: Float[TT, "time stocks"], d: int) -> Float[TT, "time stocks"]:
    """Today's value minus the value d days ago."""
    return x - delay(x, d)


@typed
def signedpower(
    x: Float[TT, "time stocks"], a: float | int
) -> Float[TT, "time stocks"]:
    """x^a preserving sign."""
    return t.sign(x) * t.abs(x).pow(float(a))


@typed
def decay_linear(x: Float[TT, "time stocks"], d: int) -> Float[TT, "time stocks"]:
    """Weighted moving average with linearly decaying weights."""
    result = t.zeros_like(x)
    weights = t.arange(1, d + 1)
    assert weights.shape == (d,)
    logger.info(f"Weights: {weights}")
    weights = weights / weights.sum()
    logger.info(f"Normalized weights: {weights}")

    for i in range(len(x)):
        start_idx = max(0, i - d + 1)
        window = x[start_idx : i + 1]
        w = weights[-len(window) :]
        result[i] = t.sum(window * w.view(-1, 1), dim=0)
    return result


@typed
def indneutralize(
    x: Float[TT, "time stocks"], g: Float[TT, "stocks"]
) -> Float[TT, "time stocks"]:
    """Cross-sectionally neutralize x within groups g."""
    result = t.zeros_like(x)
    unique_groups = t.unique(g)

    for t_idx in range(len(x)):
        for group in unique_groups:
            mask = g == group
            group_mean = x[t_idx, mask].mean()
            result[t_idx, mask] = x[t_idx, mask] - group_mean

    return result


# add test of operators here
def test_operators():
    """Test all operators with visualizations."""
    # Set random seed for reproducibility
    t.manual_seed(42)

    # Generate test data
    time, stocks = 5, 3
    x = t.randn(time, stocks)
    y = t.randn(time, stocks)
    g = t.tensor([0, 0, 1], dtype=t.float32)

    # Dictionary of operators to test
    operators = {
        "ts_max(x, 3)": lambda: ts_max(x, 3),
        "ts_min(x, 3)": lambda: ts_min(x, 3),
        "ts_argmax(x, 3)": lambda: ts_argmax(x, 3),
        "ts_argmin(x, 3)": lambda: ts_argmin(x, 3),
        "ts_rank(x, 3)": lambda: ts_rank(x, 3),
        "rank(x)": lambda: rank(x),
        "delay(x, 2)": lambda: delay(x, 2),
        "correlation(x, y, 3)": lambda: correlation(x, y, 3),
        "covariance(x, y, 3)": lambda: covariance(x, y, 3),
        "scale(x)": lambda: scale(x),
        "delta(x, 2)": lambda: delta(x, 2),
        "signedpower(x, 2)": lambda: signedpower(x, 2),
        "decay_linear(x, 3)": lambda: decay_linear(x, 3),
        "indneutralize(x, g)": lambda: indneutralize(x, g),
    }

    def plot_matrix(ax, matrix, title):
        if len(matrix.shape) == 1:  # For g which is 1D
            matrix = matrix.unsqueeze(0)
        im = ax.imshow(matrix.detach().numpy(), cmap="RdYlBu", aspect="auto")
        ax.set_title(title)

        # Add text annotations
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                text = f"{matrix[i, j]:.2f}"
                ax.text(j, i, text, ha="center", va="center")

        return im

    # Create plots for each operator
    for i, (name, op) in enumerate(operators.items()):
        plt.figure(figsize=(16, 4))

        # Plot inputs
        ax1 = plt.subplot(141)
        plot_matrix(ax1, x, "Input x")
        ax1.set_xlabel("Stocks")
        ax1.set_ylabel("Time")

        ax2 = plt.subplot(142)
        plot_matrix(ax2, y, "Input y")
        ax2.set_xlabel("Stocks")

        ax3 = plt.subplot(143)
        plot_matrix(ax3, g, "Groups g")
        ax3.set_xlabel("Stocks")

        # Plot output
        ax4 = plt.subplot(144)
        result = op()
        plot_matrix(ax4, result, f"Output: {name}")
        ax4.set_xlabel("Stocks")

        plt.tight_layout()
        plt.show()


# -----------------------------------------------------------------------------
# 101 formulaic alphas
# -----------------------------------------------------------------------------

# TODO


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
    test_operators()
