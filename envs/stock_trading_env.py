""" 
Enhanced Stock Trading Environment based on FinRL-Meta's Stock Trading Environment.
"""

import gymnasium as gym
import numpy as np
from beartype import beartype as typed
from beartype.door import die_if_unbearable as assert_type
from beartype.typing import TypeAlias
from jaxtyping import Float, Int
from numpy import ndarray as ND, random as rd

Money: TypeAlias = float
Price: TypeAlias = float
Reward: TypeAlias = float
QuantityVec: TypeAlias = Int[ND, "stock_dim"]
ActionVec: TypeAlias = Float[ND, "stock_dim"]
StateVec: TypeAlias = Float[ND, "state_dim"]

# TODO: Handle turbulence inside the agent


class StockTradingEnv(gym.Env):
    """A stock trading environment for reinforcement learning."""

    @typed
    def __init__(
        self,
        config: dict,
        gamma: float = 0.99,
        turbulence_threshold: float = 99,
        min_stock_rate: float = 0.1,
        max_stock: float = 1e2,
        initial_capital: Money = 1e6,
        buy_cost_pct: float = 1e-3,
        sell_cost_pct: float = 1e-3,
        initial_stocks: QuantityVec | None = None,
    ) -> None:
        """Initialize the stock trading environment."""
        self.price_array = config["price_array"].astype(np.float32)
        assert_type(self.price_array, Float[ND, "time_dim stock_dim"])
        self.stock_dim = self.price_array.shape[1]

        self.tech_array = config["tech_array"].astype(np.float32)
        assert_type(self.tech_array, Float[ND, "time_dim tech_dim"])
        self.tech_dim = self.tech_array.shape[1]

        self.turbulence_array = config["turbulence_array"]
        assert_type(self.turbulence_array, Float[ND, "time_dim"])

        self.turbulence_bool = (self.turbulence_array > turbulence_threshold).astype(
            np.float32
        )
        self.turbulence_array = (
            self.sigmoid_sign(self.turbulence_array, turbulence_threshold) * 2**-5
        )
        self.turbulence_array = self.turbulence_array.astype(np.float32)

        self.gamma = gamma
        self.max_stock = max_stock
        self.min_action_fraction = min_stock_rate
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.initial_capital = initial_capital
        self.initial_stocks = (
            initial_stocks
            if initial_stocks is not None
            else np.zeros(self.stock_dim, dtype=np.float32)
        )

        # Environment state variables
        self.reset()

        self.env_name = "StockEnv"
        self.state_dim = 1 + 2 + 3 * self.stock_dim + self.tech_dim
        self.action_dim = self.stock_dim
        self.max_step = self.price_array.shape[0] - 1
        self.if_discrete = False
        self.target_return = 10.0
        self.episode_return = 0.0

        self.observation_space = gym.spaces.Box(
            low=-3000, high=3000, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.action_dim,), dtype=np.float32
        )

    @typed
    def reset(self) -> StateVec:
        """Reset the environment to the initial state."""
        self.time = 0
        current_price = self.price_array[self.time]

        self.stocks = self.initial_stocks.copy()
        self.cash = self.initial_capital

        self.stocks_cool_down = np.zeros_like(self.stocks)
        self.total_asset = self.cash + np.sum(self.stocks * current_price)
        self.initial_total_asset = self.total_asset
        self.gamma_reward = 0.0

        return self._get_state(current_price)

    @typed
    def step(self, actions: ActionVec) -> tuple[StateVec, Reward, bool, dict]:
        """Take an action in the environment."""
        self.time += 1
        current_price = self.price_array[self.time]
        self.stocks_cool_down += 1

        if not self.turbulence_bool[self.time]:
            self._execute_actions(actions, current_price)
        else:
            self._handle_turbulence(current_price)

        state = self._get_state(current_price)
        new_log_total_asset = self.log_total_asset(current_price)
        reward: Reward = new_log_total_asset - self.last_log_total_asset
        self.last_log_total_asset = new_log_total_asset
        self.total_asset = total_asset

        self.gamma_reward = self.gamma_reward * self.gamma + reward
        done = self.time == self.max_step

        if done:
            reward = self.gamma_reward
            self.episode_return = total_asset / self.initial_total_asset

        return state, reward, done, {}

    @typed
    def log_total_asset(self, current_price: Price) -> float:
        """Log the total asset."""
        return np.log(max(self.cash + np.sum(self.stocks * current_price), 1.0))

    @typed
    def _execute_actions(self, actions: ActionVec, price: Price) -> None:
        """Execute buy and sell actions."""
        assert actions.shape == (self.action_dim,)
        action_quantities: QuantityVec = np.round(actions * self.max_stock).astype(int)
        min_quantity = int(self.max_stock * self.min_action_fraction)

        # Sell actions
        sell_indices = np.where(action_quantities < -min_quantity)[0]
        for idx in sell_indices:
            if price > 0:
                sell_qty = min(self.stocks[idx], -action_quantities[idx])
                self.stocks[idx] -= sell_qty
                self.cash += price * sell_qty * (1 - self.sell_cost_pct)
                self.stocks_cool_down[idx] = 0

        # Buy actions
        buy_indices = np.where(action_quantities > min_quantity)[0]
        for idx in buy_indices:
            if price > 0:
                buy_qty = min(self.cash // price, action_quantities[idx])
                self.stocks[idx] += buy_qty
                self.cash -= price * buy_qty * (1 + self.buy_cost_pct)
                self.stocks_cool_down[idx] = 0

    @typed
    def _get_state(self, price: Price) -> StateVec:
        """Get the current state of the environment."""
        cash_scaled = np.array([self.cash * 2**-12], dtype=np.float32)
        price_scaled = price * 2**-6
        stock_scaled = self.stocks / self.max_stock

        return np.hstack(
            [
                cash_scaled,
                price_scaled,
                stock_scaled,
                self.stocks_cool_down,
                self.tech_array[self.time],
            ]
        )

    @staticmethod
    def sigmoid_sign(array: np.ndarray, threshold: float) -> np.ndarray:
        """Apply a sigmoid function to the array."""
        sigmoid = lambda x: 1 / (1 + np.exp(-x * np.e)) - 0.5
        return sigmoid(array / threshold) * threshold


@typed
def f(a: Price, b: Money) -> Reward:
    return a * b


if __name__ == "__main__":
    a = Price(3.0)
    b = Price(2.0)
    c = f(a, b)
    print(c)
