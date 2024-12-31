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
TQuantityVec: TypeAlias = Float[TT, "stock_dim"]
TPriceVec: TypeAlias = Float[TT, "stock_dim"]
TActionVec: TypeAlias = Float[TT, "stock_dim"]
TStateVec: TypeAlias = Float[TT, "state_dim"]


@typed
def soft_greater(x: Float[TT, "..."], threshold: float) -> Float[TT, "..."]:
    return (threshold - x).relu().square()


class DiffStockTradingEnv(gym.Env):
    """A stock trading environment for reinforcement learning."""

    @typed
    def __init__(
        self,
        price_array: Float[TT, "time_dim stock_dim"],
        tech_array: Float[TT, "time_dim tech_dim"],
        max_stock: int,
        initial_capital: TMoney | None,
        initial_cash_ratio: float,
        buy_cost_pct: float,
        sell_cost_pct: float,
        initial_stocks: TQuantityVec,
    ) -> None:
        """Initialize environment configuration (not the state)."""
        self.price_array = price_array
        self.tech_array = tech_array
        self.stock_dim = self.price_array.shape[1]
        self.tech_dim = self.tech_array.shape[1]
        self.max_stock = max_stock
        self.initial_capital = initial_capital
        self.initial_cash_ratio = initial_cash_ratio
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.initial_stocks = initial_stocks

        self.state_dim = 2 + 3 * self.stock_dim + self.tech_dim
        self.action_dim = self.stock_dim
        self.max_step = self.price_array.shape[0] - 1
        self.env_name = "DiffStockTradingEnv"

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
        self.penalty_sum = 0.0

        # Initialize stocks
        self.stocks = self.initial_stocks.clone()

        # Initialize cash
        current_price = self.price_array[self.time]
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

    @classmethod
    @typed
    def build(
        cls,
        price_array: Float[TT, "time_dim stock_dim"],
        tech_array: Float[TT, "time_dim tech_dim"],
        max_stock: int = 100,
        initial_capital: TMoney | None = None,
        initial_cash_ratio: float = 0.5,
        buy_cost_pct: float = 1e-3,
        sell_cost_pct: float = 1e-3,
        initial_stocks: TQuantityVec | int = 100,
    ) -> "DiffStockTradingEnv":
        """User-facing factory method to create the environment."""
        if isinstance(initial_stocks, int):
            initial_stocks = t.full((price_array.shape[1],), initial_stocks).float()

        env = cls(
            price_array=price_array,
            tech_array=tech_array,
            max_stock=max_stock,
            initial_capital=initial_capital,
            initial_cash_ratio=initial_cash_ratio,
            buy_cost_pct=buy_cost_pct,
            sell_cost_pct=sell_cost_pct,
            initial_stocks=initial_stocks,
        )
        env.reset_t()
        return env

    @typed
    def subsegment(self, l: int, r: int) -> "DiffStockTradingEnv":
        """Returns a new environment based on the [l, r] subsegment of the original one."""
        assert (
            0 <= l <= r < self.price_array.shape[0]
        ), f"Invalid segment indices: [{l}, {r}]"
        env = DiffStockTradingEnv(
            price_array=self.price_array[l : r + 1],
            tech_array=self.tech_array[l : r + 1],
            max_stock=self.max_stock,
            initial_capital=self.cash,  # Use current cash as initial capital
            initial_cash_ratio=self.initial_cash_ratio,
            buy_cost_pct=self.buy_cost_pct,
            sell_cost_pct=self.sell_cost_pct,
            initial_stocks=self.stocks.clone(),  # Use current stocks as initial
        )
        env.reset_t()
        return env

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

        actual_actions, scale_penalty = self._execute_actions(actions)
        penalty = scale_penalty + self._get_penalty()
        truncated = self.penalty_sum > 100.0
        self.penalty_sum += penalty.item()
        reward = self._get_reward(actual_actions, price) - penalty
        self.last_log_total_asset = self._get_log_total_asset(price)
        state = self._get_state(price)

        terminated = self.time == self.max_step

        return state, reward, terminated, truncated, {"actual_actions": actual_actions}

    @typed
    def _get_log_total_asset(self, current_price: TPriceVec) -> TMoney:
        return self._get_total_asset(current_price).log()

    @typed
    def _get_total_asset(self, current_price: TPriceVec) -> TMoney:
        return t.maximum(
            self.cash + (self.stocks * current_price).sum(), t.tensor(1e-100)
        )

    @typed
    def _execute_actions(self, actions: TActionVec) -> tuple[TActionVec, TReward]:
        """Execute buy and sell actions. Returns the actual actions taken and the scaling penalty."""
        assert actions.shape == (self.action_dim,)
        price = self.price_array[self.time]
        action_quantities = actions * self.max_stock
        actual_prices = price * (
            1 + (actions > 0) * self.buy_cost_pct - (actions < 0) * self.sell_cost_pct
        )

        scale = t.ones(())
        qp = (action_quantities * actual_prices).sum()
        if qp > 0:
            scale = t.minimum(scale, (self.cash - 0.1) / (qp + 1e-8))
        stock_scales = t.where(
            action_quantities < 0,
            (self.stocks - 0.1) / (-action_quantities + 1e-8),
            t.tensor(1.0),
        )
        scale = t.minimum(scale, stock_scales.min())
        scale_penalty = 1.0 - scale
        if scale_penalty > 1e-8:
            logger.warning(f"Scale penalty: {scale_penalty}. Safety reset.")
            # Take a safe action
            total_value = self.cash + (self.stocks * price).sum()
            # Equal distribution target
            target_value = total_value / (2 * self.stock_dim)
            target_stocks = target_value / price
            action_quantities = target_stocks - self.stocks

        self.stocks = self.stocks + action_quantities
        self.cash = self.cash - (action_quantities * actual_prices).sum()

        return action_quantities / self.max_stock, scale_penalty

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
                self.cash.reshape(-1) / price / self.max_stock,
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

    state, _ = env.reset_t()
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
