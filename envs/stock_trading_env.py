""" 
Enhanced Stock Trading Environment based on FinRL-Meta's Stock Trading Environment.
"""

import gymnasium as gym
import numpy as np
from beartype import beartype as typed
from beartype.door import die_if_unbearable as assert_type
from beartype.typing import TypeAlias
from jaxtyping import Float, Int
from loguru import logger
from numpy import ndarray as ND, random as rd

Money: TypeAlias = float
Reward: TypeAlias = Float[ND, ""]
QuantityVec: TypeAlias = Int[ND, "stock_dim"]
PriceVec: TypeAlias = Float[ND, "stock_dim"]
ActionVec: TypeAlias = Float[ND, "stock_dim"]
StateVec: TypeAlias = Float[ND, "state_dim"]


class StockTradingEnv(gym.Env):
    """A stock trading environment for reinforcement learning."""

    @typed
    def __init__(
        self,
        price_array: Float[ND, "time_dim stock_dim"],
        tech_array: Float[ND, "time_dim tech_dim"],
        min_stock_rate: float = 0.01,
        max_stock: int = 100,
        initial_capital: Money | None = None,
        initial_cash_ratio: float = 0.5,
        buy_cost_pct: float = 1e-3,
        sell_cost_pct: float = 1e-3,
        initial_stocks: QuantityVec | int = 100,
    ) -> None:
        """Initialize the stock trading environment."""
        self.price_array = price_array.astype(np.float32)
        self.stock_dim = self.price_array.shape[1]
        self.tech_array = tech_array.astype(np.float32)
        self.tech_dim = self.tech_array.shape[1]

        self.max_stock = max_stock
        self.min_action_fraction = min_stock_rate
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.initial_stocks = (
            initial_stocks
            if not isinstance(initial_stocks, int)
            else np.full(self.stock_dim, initial_stocks, dtype=np.int64)
        ).astype(np.int64)
        stock_value = (self.price_array[0] * self.initial_stocks).sum()
        if initial_capital is None:
            r = initial_cash_ratio
            assert 0 <= r < 1
            self.initial_capital = stock_value * r / (1 - r)

        self.state_dim = 2 + 3 * self.stock_dim + self.tech_dim
        # Environment state variables
        self.reset()

        self.env_name = "StockEnv"
        self.action_dim = self.stock_dim
        self.max_step = self.price_array.shape[0] - 1
        self.if_discrete = False
        self.target_return = 10.0

        self.observation_space = gym.spaces.Box(
            low=-1e18, high=1e18, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.action_dim,), dtype=np.float32
        )

    @typed
    def reset(self, seed: int | None = None) -> tuple[StateVec, dict]:
        """Reset the environment to the initial state."""
        super().reset(seed=seed)
        self.time = 0
        current_price = self.price_array[self.time]

        self.stocks = self.initial_stocks.copy()
        self.cash = self.initial_capital

        self.stocks_cool_down = np.zeros_like(self.stocks)
        self.total_asset = self.cash + np.sum(self.stocks * current_price)
        self.initial_log_total_asset = self.log_total_asset(current_price)
        self.last_log_total_asset = self.initial_log_total_asset

        return self._get_state(current_price), {}

    @typed
    def get_reward(self, actions: ActionVec, price: PriceVec) -> Reward:
        """Get the reward for the current state."""
        new_log_total_asset = self.log_total_asset(price)
        reward = np.array(
            new_log_total_asset - self.last_log_total_asset, dtype=np.float32
        )
        return reward

    @typed
    def step(self, actions: ActionVec) -> tuple[StateVec, Reward, bool, bool, dict]:
        """Take an action in the environment."""
        self.time += 1
        current_price = self.price_array[self.time]
        self.stocks_cool_down += 1

        self._execute_actions(actions, current_price)

        state = self._get_state(current_price)
        reward = self.get_reward(actions, current_price)

        self.last_log_total_asset = self.log_total_asset(current_price)
        terminated = self.time == self.max_step
        truncated = False

        return state, reward, terminated, truncated, {}

    @typed
    def log_total_asset(self, current_price: PriceVec) -> float:
        """Log the total asset."""
        return np.log(max(self.cash + np.sum(self.stocks * current_price), 1.0))

    @typed
    def _execute_actions(self, actions: ActionVec, price: PriceVec) -> None:
        """Execute buy and sell actions."""
        assert actions.shape == (self.action_dim,)
        action_quantities: QuantityVec = np.round(actions * self.max_stock).astype(int)
        min_quantity = int(self.max_stock * self.min_action_fraction)

        # Sell actions
        sell_indices = np.where(action_quantities < -min_quantity)[0]
        for idx in sell_indices:
            if price[idx] > 0:
                sell_qty = min(self.stocks[idx], -action_quantities[idx])
                self.stocks[idx] -= sell_qty
                self.cash += price[idx] * sell_qty * (1 - self.sell_cost_pct)
                self.stocks_cool_down[idx] = 0
            else:
                logger.warning(f"Price is negative for stock {idx}")

        # Buy actions
        buy_indices = np.where(action_quantities > min_quantity)[0]
        for idx in buy_indices:
            if price[idx] > 0:
                buy_qty = min(self.cash // price[idx], action_quantities[idx])
                self.stocks[idx] += buy_qty
                self.cash -= price[idx] * buy_qty * (1 + self.buy_cost_pct)
                self.stocks_cool_down[idx] = 0
            else:
                logger.warning(f"Price is negative for stock {idx}")

    @typed
    def get_cash_ratio(self, price: PriceVec) -> float:
        return self.cash / np.exp(self.log_total_asset(price))

    @typed
    def _get_state(self, price: PriceVec) -> StateVec:
        """Get the current state of the environment."""
        cash_np = np.array([self.get_cash_ratio(price)], dtype=np.float32)
        stocks_float = self.stocks.astype(np.float32)
        cooldown_float = self.stocks_cool_down.astype(np.float32)
        max_stock_np = np.array([self.max_stock], dtype=np.float32)

        state = np.hstack(
            [
                cash_np,
                price,
                stocks_float,
                cooldown_float,
                self.tech_array[self.time],
                max_stock_np,
            ]
        )
        assert state.shape == (self.state_dim,)
        return state


def test_env_1():
    """Rewards should be positive and slowly decreasing in the first phase and in the middle, then zero."""
    n = 20
    prices = np.stack(
        [
            np.linspace(1, 1001, n),
            np.linspace(1, 2001, n),
        ],
        axis=1,
    )
    techs = np.stack(
        [
            np.linspace(0, 1, n),
            np.linspace(0, 1, n),
        ],
        axis=1,
    )
    env = StockTradingEnv(prices, techs, initial_stocks=1)
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


if __name__ == "__main__":
    test_env_1()
