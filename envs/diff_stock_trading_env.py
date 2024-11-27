import gymnasium as gym
import numpy as np
import torch as t
from beartype import beartype as typed
from beartype.door import die_if_unbearable as assert_type
from beartype.typing import TypeAlias
from envs.stock_trading_env import ActionVec, Money, Reward, StateVec
from jaxtyping import Float, Int
from loguru import logger
from numpy import ndarray as ND, random as rd
from torch import Tensor as TT

TMoney: TypeAlias = Float[TT, ""]
TReward: TypeAlias = Float[TT, ""]
TQuantityVec: TypeAlias = Int[TT, "stock_dim"]
TPriceVec: TypeAlias = Float[TT, "stock_dim"]
TActionVec: TypeAlias = Float[TT, "stock_dim"]
TStateVec: TypeAlias = Float[TT, "state_dim"]


@typed
def soft_greater(x: Float[TT, ""], threshold: float) -> Float[TT, ""]:
    return (threshold - x).relu().square()


class DiffStockTradingEnv(gym.Env):
    """A stock trading environment for reinforcement learning."""

    @typed
    def __init__(
        self,
        price_array: Float[TT, "time_dim stock_dim"],
        tech_array: Float[TT, "time_dim tech_dim"],
        max_stock: int = 100,
        initial_capital: TMoney | None = None,
        initial_cash_ratio: float = 0.5,
        buy_cost_pct: float = 1e-3,
        sell_cost_pct: float = 1e-3,
        initial_stocks: TQuantityVec | int = 100,
    ) -> None:
        """Initialize the stock trading environment."""
        self.price_array = price_array
        self.stock_dim = self.price_array.shape[1]
        self.tech_array = tech_array
        self.tech_dim = self.tech_array.shape[1]

        self.max_stock = max_stock
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.initial_stocks = (
            initial_stocks
            if not isinstance(initial_stocks, int)
            else t.full((self.stock_dim,), initial_stocks)
        ).float()
        self.initial_capital = initial_capital
        self.initial_cash_ratio = initial_cash_ratio

        self.state_dim = 2 + 3 * self.stock_dim + self.tech_dim
        # Environment state variables
        self.reset()

        self.env_name = "DiffStockTradingEnv"
        self.action_dim = self.stock_dim
        self.max_step = self.price_array.shape[0] - 1

        self.observation_space = gym.spaces.Box(
            low=-1e18, high=1e18, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.action_dim,), dtype=np.float32
        )

    @typed
    def reset_t(self, seed: int | None = None) -> tuple[TStateVec, dict]:
        """Reset the environment to the initial state."""
        super().reset(seed=seed)
        self.time = 0
        current_price = self.price_array[self.time]
        self.stocks = self.initial_stocks.clone()
        stock_value = (current_price * self.stocks).sum()
        if self.initial_capital is None:
            r = self.initial_cash_ratio
            assert 0 <= r < 1
            self.cash = stock_value * r / (1 - r)
            assert abs(self.cash / (self.cash + stock_value) - r) < 1e-6
        else:
            self.cash = self.initial_capital
        self.last_log_total_asset = self._get_log_total_asset(current_price)

        return self._get_state(current_price), {}

    @typed
    def _get_reward(self, actions: TActionVec, price: TPriceVec) -> TReward:
        """Get the reward for the current state."""
        new_log_total_asset = self._get_log_total_asset(price)
        reward = new_log_total_asset - self.last_log_total_asset
        return reward

    @typed
    def step_t(
        self, actions: TActionVec
    ) -> tuple[TStateVec, TReward, bool, bool, dict]:
        """Take an action in the environment."""
        self.time += 1
        price = self.price_array[self.time]

        actions = t.where(t.isfinite(actions), actions, t.tensor(0.0))
        actions.data.clamp_(-1, 1)

        self._execute_actions(actions)
        penalty = self._get_penalty()
        reward = self._get_reward(actions, price) - penalty
        self.last_log_total_asset = self._get_log_total_asset(price)
        state = self._get_state(price)

        terminated = self.time == self.max_step
        truncated = (penalty > 1).item()

        return state, reward, terminated, truncated, {}

    @typed
    def _get_log_total_asset(self, current_price: TPriceVec) -> TMoney:
        return self._get_total_asset(current_price).log()

    @typed
    def _get_total_asset(self, current_price: TPriceVec) -> TMoney:
        return t.maximum(
            self.cash + (self.stocks * current_price).sum(), t.tensor(1e-100)
        )

    @typed
    def _execute_actions(self, actions: TActionVec) -> None:
        """Execute buy and sell actions."""
        assert actions.shape == (self.action_dim,)
        price = self.price_array[self.time]
        action_quantities = actions * self.max_stock
        actual_prices = price * (
            1 + (actions > 0) * self.buy_cost_pct - (actions < 0) * self.sell_cost_pct
        )
        self.stocks = self.stocks + action_quantities
        self.cash = self.cash - (action_quantities * actual_prices).sum()

    @typed
    def _get_penalty(self) -> TReward:
        cash_ratio = self._get_cash_ratio(self.price_array[self.time])
        penalty = soft_greater(cash_ratio, 0.1) + soft_greater(1 - cash_ratio, 0.1)
        return penalty

    @typed
    def _get_cash_ratio(self, price: TPriceVec) -> Float[TT, ""]:
        raw = self.cash / self._get_total_asset(price)
        clipped = t.clip(raw, -1.0, 2.0)
        if not clipped.isfinite().all():
            logger.warning(f"Cash ratio is NaN: {raw}")
            return t.zeros_like(raw)
        return clipped

    @typed
    def _get_state(self, price: TPriceVec) -> TStateVec:
        """Get the current state of the environment."""
        cash_ratio = self._get_cash_ratio(price).reshape(1)

        state = t.cat(
            [
                cash_ratio,
                price,
                self.stocks / self.max_stock,
                self.tech_array[self.time],
                self.cash.reshape(1) / self.max_stock,
                self.cash.reshape(1) / price / self.max_stock,
            ]
        )
        max_abs = state.abs().max()

        MAX = 1e6
        if max_abs > MAX:
            logger.warning(f"State max abs: {max_abs}")
            state.clip_(-MAX, MAX)
        all_good = (state.abs() < MAX).all() and state.isfinite().all()
        if not all_good:
            logger.warning(f"State is bad: {state}")

        # to detach or not to detach?
        # state = state.detach()

        assert state.shape == (self.state_dim,)
        return state

    @typed
    def reset(self, seed: int | None = None) -> tuple[StateVec, dict]:
        state, info = self.reset_t(seed)
        return state.numpy(), info

    @typed
    def step(self, actions: ActionVec) -> tuple[StateVec, Reward, bool, bool, dict]:
        actions = t.tensor(actions)
        state, reward, terminated, truncated, info = self.step_t(actions)
        return state.numpy(), reward.numpy(), terminated, truncated, info


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
