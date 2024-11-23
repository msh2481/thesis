import numpy as np
from beartype import beartype as typed
from envs.stock_trading_env import (
    ActionVec,
    PriceVec,
    Reward,
    StateVec,
    StockTradingEnv,
)


class CashRatioEnv(StockTradingEnv):
    """
    Environment where the goal is to keep the cash ratio close to 0.5 while doing as much trading as possible.
    """

    @classmethod
    @typed
    def create(cls, n_stocks: int, tech_per_stock: int, n_steps: int):
        n_tech = n_stocks * tech_per_stock
        price_array = np.ones((n_steps, n_stocks))
        tech_array = np.random.randn(n_steps, n_tech)
        return cls(price_array, tech_array, initial_cash_ratio=0.4)

    @typed
    def step(self, actions: ActionVec) -> tuple[StateVec, Reward, bool, bool, dict]:
        """Take an action in the environment."""
        self.time += 1
        current_price = self.price_array[self.time]
        self.stocks_cool_down += 1

        old_ratio = self.get_cash_ratio(current_price)
        self._execute_actions(actions, current_price)
        new_ratio = self.get_cash_ratio(current_price)

        state = self._get_state(current_price)
        old_new = np.abs(old_ratio - new_ratio)
        old_dev = np.abs(0.5 - old_ratio)
        new_dev = np.abs(0.5 - new_ratio)
        reward = np.array(old_new**2 - (old_dev**2 + new_dev**2))

        self.last_log_total_asset = self.log_total_asset(current_price)
        terminated = self.time == self.max_step
        truncated = False

        return state, reward, terminated, truncated, {}


class PredictableEnv(StockTradingEnv):
    """
    Environment where the next price is one of the signals.
    """

    @classmethod
    @typed
    def build_arrays(cls, n_steps: int, n_stocks: int, tech_per_stock: int):
        n_tech = n_stocks * tech_per_stock
        price_array = np.random.randn(n_steps, n_stocks)
        price_array = np.cumsum(price_array, axis=0) / np.sqrt(n_steps)
        price_array = np.exp(price_array)
        tech_array = np.random.randn(n_steps, n_tech)
        for i in range(n_stocks):
            tech_array[:, i * tech_per_stock] = np.roll(price_array[:, i], -1)

        # check that the next price is one of the signals
        for t in range(n_steps - 1):
            for i in range(n_stocks):
                assert (
                    np.abs(price_array[t + 1, i] - tech_array[t, i * tech_per_stock])
                    < 1e-6
                )
        return price_array, tech_array

    @classmethod
    @typed
    def create(cls, n_stocks: int, tech_per_stock: int, n_steps: int):
        assert tech_per_stock > 0
        n_tech = n_stocks * tech_per_stock
        price_array, tech_array = cls.build_arrays(n_steps, n_stocks, tech_per_stock)
        instance = cls(price_array, tech_array, initial_cash_ratio=0.5)
        instance.tech_per_stock = tech_per_stock
        instance.n_steps = n_steps
        return instance

    @typed
    def reset(self, seed: int | None = None) -> tuple[StateVec, dict]:
        n_steps = len(self.price_array)
        n_stocks = self.stock_dim
        tech_per_stock = self.tech_dim // n_stocks
        self.price_array, self.tech_array = self.build_arrays(
            n_steps, n_stocks, tech_per_stock
        )
        return super().reset(seed)


if __name__ == "__main__":
    env = PredictableEnv.create(2, 1, 100)
