import gymnasium as gym
import numpy as np
import torch as t
from beartype import beartype as typed
from jaxtyping import Float, Int, jaxtyped
from numpy import ndarray as ND
from torch import Tensor as TT

FEE_PCT = 1e-3

Features = (
    Float[ND, "feature_dim"]
    | Float[TT, "feature_dim"]
    | Float[ND, "stock_dim feature_dim"]
    | Float[TT, "stock_dim feature_dim"]
)


@typed
def logits_to_position(logits: Float[TT, "stock_dim"]) -> Float[TT, "stock_dim"]:
    unnormalized = logits.clamp(-10, 10).exp()
    return unnormalized / (unnormalized.sum() + 1)


@typed
def position_to_logits(position: Float[TT, "stock_dim"]) -> Float[TT, "stock_dim"]:
    s = position.sum()
    assert s <= (1 + 1e-4), f"Position sum: {s}"
    position = (position + 1e-8) / max(1 - s, 1e-8)
    return position.log()


def test_logits():
    x = t.tensor([8.0, 9, 10])
    y = logits_to_position(x)
    print(y)
    z = position_to_logits(y)
    print(z)


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
        position = self.position()
        logits = position_to_logits(position)
        assert not t.isnan(logits).any(), "NaN logits"
        features = t.cat([self.prices[:, None], logits[:, None], self.tech], dim=1)
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
        fee = (position * value - self.stocks * self.prices).abs() * FEE_PCT
        new_value = value - fee.sum()
        self.stocks = position * new_value / self.prices
        self.cash = (1 - position.sum()) * new_value

    @typed
    def set_prices_and_tech(
        self, prices: Float[TT, "stock_dim"], tech: Float[TT, "stock_dim tech_dim"]
    ) -> None:
        assert prices.shape == (self.stock_dim,)
        self.prices = prices
        self.tech = tech

    @typed
    def detach(self) -> "State":
        return State(
            cash=self.cash.detach(),
            stocks=self.stocks.detach(),
            tech=self.tech.detach(),
            prices=self.prices.detach(),
        )


@jaxtyped(typechecker=typed)
def compute_optimal_1d(
    prices: Float[TT, "time_dim"],
) -> tuple[Float[TT, ""], Float[TT, "time_dim-1"]]:
    n = len(prices)
    # dp[prefix, position] = max amount of cash (for 0) /stock (for 1) in such state
    # it is before we traded on prices[prefix]
    dp = t.full((n + 1, 2), -1e18, dtype=t.float32)
    pr = t.full((n + 1, 2), -1, dtype=t.int32)
    dp[0, 0] = 1.0
    for i in range(n):
        sell_price = prices[i] * (1 - FEE_PCT)
        buy_price = prices[i] * (1 + FEE_PCT)
        updates = [
            (0, 0, 1.0),  # stay zero
            (1, 1, 1.0),  # stay long
            (0, 1, sell_price),  # sell
            (1, 0, 1 / buy_price),  # buy
        ]
        for np, op, v in updates:
            if dp[i + 1, np] < dp[i, op] * v:
                dp[i + 1, np] = dp[i, op] * v
                pr[i + 1, np] = op
    value = dp[n, 0]
    positions = []
    p = 0
    i = n
    while i >= 0:
        positions.append(p)
        p = pr[i, p]
        i -= 1
    return value, t.tensor(positions[::-1][1:-1], dtype=t.float32)


class DiffStockTradingEnv(gym.Env):
    """A stock trading environment for reinforcement learning."""

    @typed
    def __init__(
        self,
        price_array: Float[TT, "time_dim stock_dim"],
        tech_array: Float[TT, "time_dim stock_dim tech_dim"],
        initial_value: Float[TT, ""] = t.ones(()),
        initial_position: Float[TT, "stock_dim"] | None = None,
        numpy: bool = False,
        flat: bool = False,
    ) -> None:
        self.price_array = price_array
        self.tech_array = tech_array
        self.n_steps, self.stock_dim, self.tech_dim = self.tech_array.shape
        self.initial_value = initial_value
        self.initial_position = (
            initial_position
            if initial_position is not None
            else t.ones(self.stock_dim) / (self.stock_dim + 1)
        )
        self.numpy = numpy
        self.flat = flat
        self.env_name = "DiffStockTradingEnv"

        self.state, _ = self.reset_state()
        self.state_shape = self.state.features(numpy=self.numpy, flat=self.flat).shape

        self.observation_space = gym.spaces.Box(
            low=-1e18,
            high=1e18,
            shape=self.state_shape,
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
        return state.features(numpy=self.numpy, flat=self.flat), info

    @typed
    def get_optimal_positions(self) -> Float[TT, "time_dim stock_dim"]:
        results = []
        for i in range(self.stock_dim):
            results.append(compute_optimal_1d(self.price_array[:, i])[1])
        result = t.stack(results, dim=1)
        return result / self.stock_dim

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
        self.state.set_prices_and_tech(
            self.price_array[self.time], self.tech_array[self.time]
        )

        new_log_value = self.state.value().log()
        reward = new_log_value - self.last_log_value
        self.last_log_value = new_log_value

        truncated = False
        terminated = self.time == self.n_steps - 1

        return self.state, reward, terminated, truncated, {}

    @typed
    def step(
        self, actions: Float[TT, "stock_dim"]
    ) -> tuple[Features, Float[TT, ""] | Float[ND, ""], bool, bool, dict]:
        state, reward, terminated, truncated, info = self.make_step(actions)
        return (
            state.features(numpy=self.numpy, flat=self.flat),
            reward.numpy() if self.numpy else reward,
            terminated,
            truncated,
            info,
        )


def test_env_1():
    """Rewards should be positive and slowly decreasing in the first phase, then zero."""
    n = 20
    prices = t.stack(
        [
            t.linspace(1, 1001, n),
            t.linspace(1, 2001, n),
        ],
        dim=1,
    )
    techs = t.stack(
        [
            t.linspace(0, 1, n).unsqueeze(1),
            t.linspace(0, 1, n).unsqueeze(1),
        ],
        dim=1,
    )
    env = DiffStockTradingEnv(prices, techs, initial_value=t.tensor(1.0))
    state, _ = env.reset_state()

    for i in range(10):
        action = t.zeros(env.stock_dim)
        action[0] = (i + 1) / 10
        state, reward, terminated, truncated, _ = env.make_step(action)
        done = terminated or truncated
        print(reward.item(), done)

    print("---")
    for _ in range(5):
        action = t.zeros(env.stock_dim)
        state, reward, terminated, truncated, _ = env.make_step(action)
        done = terminated or truncated
        print(reward.item(), done)


def test_env_2():
    """Test position setting with constant prices."""
    n = 50
    prices = t.ones((n, 2))  # Constant prices to focus on quantities
    techs = t.zeros((n, 2, 1))  # No technical indicators needed
    env = DiffStockTradingEnv(
        prices,
        techs,
        initial_value=t.tensor(1.0),
        initial_position=t.tensor([0.3, 0.3]),  # 40% cash initially
    )

    # Track history
    cash_history = []
    stock_history = []
    position_history = []

    state, _ = env.reset_state()
    cash_history.append(state.cash.item())
    stock_history.append(state.stocks[0].item())
    position_history.append(state.position()[0].item())

    # Try different position settings
    for _ in range(n - 1):
        new_position = t.tensor([0.4, 0.4])  # Try to increase stock positions
        state, reward, terminated, truncated, _ = env.make_step(new_position)

        cash_history.append(state.cash.item())
        stock_history.append(state.stocks[0].item())
        position_history.append(state.position()[0].item())

        if terminated or truncated:
            break

    # Plot results
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(cash_history, "b-", label="Cash")
    plt.plot(stock_history, "r-", label="Stock 1 Quantity")
    plt.grid(True)
    plt.legend()
    plt.title("Cash and Stock Quantities")

    plt.subplot(2, 1, 2)
    plt.plot(position_history, "g-", label="Stock 1 Position")
    plt.grid(True)
    plt.legend()
    plt.title("Position History")

    plt.tight_layout()
    plt.savefig("logs/position_test.png")
    plt.close()


if __name__ == "__main__":
    test_logits()
