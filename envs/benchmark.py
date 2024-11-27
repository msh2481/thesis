import numpy as np
import numpy.random as rd
import seaborn as sns
import torch as t
from beartype import beartype as typed
from envs.diff_stock_trading_env import (
    DiffStockTradingEnv,
    TActionVec,
    TReward,
    TStateVec,
)
from envs.stock_trading_env import (
    ActionVec,
    PriceVec,
    Reward,
    StateVec,
    StockTradingEnv,
)
from jaxtyping import Float, Int
from matplotlib import pyplot as plt
from torch import Tensor as TT

sns.set_theme(context="notebook", style="white")


class CashRatioEnv(DiffStockTradingEnv):
    """
    Environment where the goal is to keep the cash ratio close to 0.5 while doing as much trading as possible.
    """

    @classmethod
    @typed
    def create(cls, n_stocks: int, tech_per_stock: int, n_steps: int):
        n_tech = n_stocks * tech_per_stock
        price_array = t.ones((n_steps, n_stocks))
        tech_array = t.randn((n_steps, n_tech))
        return cls(price_array, tech_array, initial_cash_ratio=0.4)

    @typed
    def step_t(
        self, actions: TActionVec
    ) -> tuple[TStateVec, TReward, bool, bool, dict]:
        """Take an action in the environment."""
        self.time += 1
        price = self.price_array[self.time]

        old_ratio = self._get_cash_ratio(price)
        self._execute_actions(actions)
        new_ratio = self._get_cash_ratio(price)
        old_new = t.abs(old_ratio - new_ratio)
        old_dev = t.abs(0.5 - old_ratio)
        new_dev = t.abs(0.5 - new_ratio)
        reward = old_new**2 - (old_dev**2 + new_dev**2)
        self.last_log_total_asset = self._get_log_total_asset(price)

        state = self._get_state(price)

        terminated = self.time == self.max_step
        truncated = False

        return state, reward, terminated, truncated, {}


class PredictableEnv(DiffStockTradingEnv):
    """
    Environment where the next price is one of the signals.
    """

    dt = 5

    @classmethod
    @typed
    def build_arrays(cls, n_steps: int, n_stocks: int, tech_per_stock: int):
        n_tech = n_stocks * tech_per_stock
        # price_array = t.tensor(
        #     [[1 if i % 2 == 0 else -1 for j in range(n_stocks)] for i in range(n_steps)]
        # )
        price_array = t.randn((n_steps, n_stocks)) / n_steps**0.5
        price_array = t.cumsum(price_array, dim=0)
        price_array = t.exp(price_array)
        tech_array = t.randn((n_steps, n_tech))
        for i in range(n_stocks):
            tech_array[:, i * tech_per_stock] = t.roll(price_array[:, i], -cls.dt)

        # check that the next price is one of the signals
        for tt in range(n_steps - cls.dt):
            for i in range(n_stocks):
                assert (
                    t.abs(
                        price_array[tt + cls.dt, i] - tech_array[tt, i * tech_per_stock]
                    ).item()
                    < 1e-6
                )
        return price_array, tech_array

    @classmethod
    @typed
    def create(cls, n_stocks: int, tech_per_stock: int, n_steps: int):
        assert tech_per_stock > 0
        price_array, tech_array = cls.build_arrays(n_steps, n_stocks, tech_per_stock)
        instance = cls(price_array, tech_array, initial_cash_ratio=0.5)
        instance.tech_per_stock = tech_per_stock
        instance.n_steps = n_steps
        return instance

    @typed
    def reset_t(self, seed: int | None = None) -> tuple[TStateVec, dict]:
        n_steps = len(self.price_array)
        n_stocks = self.stock_dim
        tech_per_stock = self.tech_dim // n_stocks
        self.price_array, self.tech_array = self.build_arrays(
            n_steps, n_stocks, tech_per_stock
        )
        return super().reset_t(seed)


def test_ground_truth():
    env = PredictableEnv.create(1, 1, 100)
    state, _ = env.reset()
    cashes = []
    actions = []
    prices = []
    stocks = []
    techs = []
    rewards = []
    for _ in range(10):
        cash_ratio, price, stock, tech = state
        cash = env.cash
        action = np.array([((price > tech).astype(np.float32) - stock / env.max_stock)])
        state, reward, terminated, truncated, info = env.step(action)
        cashes.append(cash)
        actions.append(action)
        prices.append(price)
        stocks.append(stock)
        techs.append(tech)
        rewards.append(reward)
        done = terminated or truncated
        if done:
            break

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 3, 1)
    plt.plot(cashes, "b-", label="Cash")
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(prices, "r-", label="Price")
    plt.plot(techs, "m-", label="Technical Signal")
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.plot(stocks, "g-", label="Stock Holdings")
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.plot(actions, "k-", label="Action")
    plt.axhline(0, color="k", linestyle="--")
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(rewards, "y-", label="Reward")
    plt.plot(np.cumsum(rewards), "r-", label="Cumulative Reward")
    plt.axhline(0, color="k", linestyle="--")
    plt.legend()

    plt.tight_layout()
    plt.show()


def generate_weird_tensor(n: int) -> Float[TT, "n"]:
    values = [t.inf, -t.inf, t.nan, -t.nan, 0.0, 1e-1000, 1e-100, 1e-20, 1e20, 1e100]
    return t.tensor(rd.choice(values, size=(n,)))


def assert_good_tensor(tensor: Float[TT, "n"]):
    assert not t.isnan(tensor).any()
    assert not t.isinf(tensor).any()
    assert t.isfinite(tensor).all()


def test_corner_cases():
    env = PredictableEnv.create(2, 2, 100)
    state, _ = env.reset_t()
    assert_good_tensor(state)
    for _ in range(10):
        action = generate_weird_tensor(env.action_dim)
        state, reward, terminated, truncated, _ = env.step_t(action)
        done = terminated or truncated
        print(state, reward)
        assert_good_tensor(state)
        assert_good_tensor(reward)


if __name__ == "__main__":
    test_corner_cases()
