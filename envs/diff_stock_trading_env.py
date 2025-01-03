import gymnasium as gym
import numpy as np
import torch as t
from beartype import beartype as typed
from jaxtyping import Float, Int
from numpy import ndarray as ND
from torch import Tensor as TT

Features = (
    Float[ND, "feature_dim"]
    | Float[TT, "feature_dim"]
    | Float[ND, "stock_dim feature_dim"]
    | Float[TT, "stock_dim feature_dim"]
)


class State:
    cash: Float[TT, ""]
    stocks: Float[TT, "stock_dim"]
    tech: Float[TT, "stock_dim tech_dim"]
    prices: Float[TT, "stock_dim"]

    @typed
    def __init__(
        self,
        cash: Float[TT, ""],
        stocks: Float[TT, "stock_dim"],
        tech: Float[TT, "stock_dim tech_dim"],
        prices: Float[TT, "stock_dim"],
    ):
        self.stock_dim, self.tech_dim = tech.shape

        self.cash = cash
        self.stocks = stocks
        assert stocks.shape == (self.stock_dim,)
        self.tech = tech
        assert tech.shape == (self.stock_dim, self.tech_dim)
        self.prices = prices
        assert prices.shape == (self.stock_dim,)

    @typed
    def value(self) -> Float[TT, ""]:
        return self.cash + (self.stocks * self.prices).sum()

    @typed
    def position(self) -> Float[TT, "stock_dim"]:
        stock_values = self.stocks * self.prices
        return stock_values / (self.cash + stock_values.sum())

    @typed
    def features(self, numpy: bool = False, flat: bool = False) -> Features:
        features = t.cat([self.prices, self.position(), self.tech], dim=1)
        if flat:
            features = features.flatten()
        if numpy:
            features = features.numpy()
        return features

    @typed
    def set_position(self, position: Float[TT, "stock_dim"]) -> None:
        assert position.shape == (self.stock_dim,)
        assert position.sum() <= 1.0, f"Position sum: {position.sum()}"
        assert position.min() >= 0.0, f"Position min: {position.min()}"
        value = self.value()
        self.stocks = position * value / self.prices
        self.cash = (1 - position.sum()) * value

    @typed
    def set_prices(self, prices: Float[TT, "stock_dim"]) -> None:
        assert prices.shape == (self.stock_dim,)
        self.prices = prices

    @typed
    def detach(self) -> "State":
        return State(
            cash=self.cash.detach(),
            stocks=self.stocks.detach(),
            tech=self.tech.detach(),
            prices=self.prices.detach(),
        )


class DiffStockTradingEnv(gym.Env):
    """A stock trading environment for reinforcement learning."""

    @typed
    def __init__(
        self,
        price_array: Float[TT, "time_dim stock_dim"],
        tech_array: Float[TT, "time_dim stock_dim tech_dim"],
        initial_value: Float[TT, ""] = t.ones(()),
        initial_position: Float[TT, "stock_dim"] | None = None,
        buy_cost_pct: float = 1e-3,
        sell_cost_pct: float = 1e-3,
        numpy: bool = False,
        flat: bool = False,
    ) -> None:
        self.price_array = price_array
        self.tech_array = tech_array
        self.n_steps, self.stock_dim, self.tech_dim = self.tech_array.shape
        self.initial_value = initial_value
        self.initial_position = initial_position or t.ones(()) / (self.stock_dim + 1)
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.numpy = numpy
        self.flat = flat
        self.env_name = "DiffStockTradingEnv"

        self.state, _ = self.reset()

        self.observation_space = gym.spaces.Box(
            low=-1e18,
            high=1e18,
            shape=self.state.shape,
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(self.stock_dim,),
            dtype=np.float32,
        )

    @typed
    def reset_state(self, seed: int | None = None) -> tuple[State, dict]:
        """Reset the environment to the initial state."""
        super().reset(seed=seed)
        self.time = 0
        self.penalty_sum = 0.0

        self.state = State(
            cash=self.initial_value,
            stocks=t.zeros(self.stock_dim),
            tech=self.tech_array[self.time],
            prices=self.price_array[self.time],
        )
        self.state.set_position(self.initial_position)

        self.last_log_value = self.state.value().log()

        # to detach or not to detach?
        # state = state.detach()

        return self.state, {}

    @typed
    def reset(self, seed: int | None = None) -> tuple[Features, dict]:
        state, info = self.reset_state(seed)
        return state.features(numpy=True, flat=self.flat), info

    @typed
    def subsegment(self, l: int, r: int) -> "DiffStockTradingEnv":
        """Returns a new environment based on the [l, r] subsegment of the original one."""
        assert (
            0 <= l <= r < self.price_array.shape[0]
        ), f"Invalid segment indices: [{l}, {r}]"
        return DiffStockTradingEnv(
            price_array=self.price_array[l : r + 1],
            tech_array=self.tech_array[l : r + 1],
            initial_value=self.initial_value,
            initial_position=self.initial_position,
            buy_cost_pct=self.buy_cost_pct,
            sell_cost_pct=self.sell_cost_pct,
            numpy=self.numpy,
            flat=self.flat,
        )

    @typed
    def make_step(
        self, new_position: Float[TT, "stock_dim"]
    ) -> tuple[State, Float[TT, ""], bool, bool, dict]:
        """Take an action in the environment."""
        self.time += 1
        self.state.set_position(new_position)
        self.state.set_prices(self.price_array[self.time])

        new_log_value = self.state.value().log()
        reward = new_log_value - self.last_log_value
        self.last_log_value = new_log_value

        truncated = False
        terminated = self.time == self.max_step

        return self.state, reward, terminated, truncated, {}

    @typed
    def step(
        self, actions: Float[TT, "stock_dim"]
    ) -> tuple[Features, Float[TT, ""] | Float[ND, ""], bool, bool, dict]:
        state, reward, terminated, truncated, info = self.make_step(actions)
        return (
            state.features(numpy=True, flat=self.flat),
            reward.numpy() if self.numpy else reward,
            terminated,
            truncated,
            info,
        )


def test_env_1():
    """Rewards should be positive and slowly decreasing in the first phase and in the middle, then zero."""
    n = 20
    prices = t.stack(
        [
            t.linspace(1, 1001, n),
            t.linspace(1, 2001, n),
        ],
        axis=1,
    )
    techs = t.stack(
        [
            t.linspace(0, 1, n),
            t.linspace(0, 1, n),
        ],
        axis=1,
    )
    env = DiffStockTradingEnv(prices, techs, initial_stocks=1)
    state, _ = env.reset()
    for _ in range(10):
        action = np.zeros(env.action_dim)
        _, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        print(reward, done)
    action = -np.ones(env.action_dim)
    _, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    print("---")
    print(reward, done)
    print("---")
    for _ in range(5):
        action = np.zeros(env.action_dim)
        _, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        print(reward, done)


def test_env_2():
    """Test safety mechanism by continuously buying first stock."""
    n = 50
    prices = t.ones((n, 2)) * 1.0  # Constant prices to focus on quantities
    techs = t.zeros((n, 2))  # No technical indicators needed
    env = DiffStockTradingEnv.build(
        prices, techs, initial_stocks=1, initial_cash_ratio=0.4
    )

    # Track history
    cash_history = []
    stock_history = []
    action_history = []
    actual_action_history = []

    state, _ = env.reset_state()
    cash_history.append(env.cash.item())
    stock_history.append(env.stocks[0].item())

    # Try to buy first stock aggressively
    for _ in range(n - 1):
        action = t.tensor([0.0007, 0.0])  # Always try to buy max of first stock
        action_history.append(action[0].item())

        state, reward, terminated, truncated, info = env.step_t(action)
        actual_action_history.append(info["actual_actions"][0].item())
        cash_history.append(env.cash.item())
        stock_history.append(env.stocks[0].item())

        if terminated or truncated:
            break

    # Plot results
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(cash_history, "b-", label="Cash")
    plt.plot(stock_history, "r-", label="Stock 1 Quantity")
    plt.axhline(0.1, color="k", linestyle="--", label="Safety Threshold")
    plt.grid(True)
    plt.legend()
    plt.title("Cash and Stock Quantities")

    plt.subplot(2, 1, 2)
    plt.plot(action_history, "b-", label="Attempted Actions")
    plt.plot(actual_action_history, "r-", label="Actual Actions")
    plt.grid(True)
    plt.legend()
    plt.title("Actions vs Actual Actions")

    plt.tight_layout()
    plt.savefig("logs/safety_test.png")
    plt.close()


if __name__ == "__main__":
    test_env_2()
