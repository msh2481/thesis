from collections import defaultdict

import matplotlib.pyplot as plt

import numpy as np
import torch as t
from beartype import beartype as typed
from beartype.typing import Callable, Iterator
from data_processing import get_data, macd, rsi_14
from envs.stock_trading_env import StateVec, StockTradingEnv
from gymnasium import Env
from jaxtyping import Float, Int
from loguru import logger
from models import Actor, HODL, StateAdapter
from numpy import ndarray as ND
from torch import nn, Tensor as TT
from torch.distributions import Distribution
from tqdm.auto import tqdm


@typed
def evaluate_policy(
    policy: Actor,
    env: Env,
    num_episodes: int = 100,
    observers: dict[str, Callable[[StateVec, float, float], float]] = {},
) -> tuple[float, dict[str, float], Float[ND, "num_episodes num_steps"]]:
    total_reward = 0
    all_totals = []
    all_metrics = defaultdict(list)
    for _ in tqdm(range(num_episodes), desc="Evaluating policy"):
        current_totals = []
        current_total = 0
        current_metrics = defaultdict(list)
        state = env.reset()
        done = False
        while not done:
            action = policy.actor(state).sample().numpy()
            next_state, reward, done, _ = env.step(action)
            current_total += reward
            state = next_state
            current_totals.append(current_total)
            for observer_name, observer in observers.items():
                current_metrics[observer_name].append(
                    observer(state, reward, current_total)
                )
        total_reward += current_total
        all_totals.append(current_totals)
        for observer_name in observers:
            all_metrics[observer_name].append(current_metrics[observer_name])
    histories = np.array(all_totals)

    for observer_name in observers:
        list_of_lists = all_metrics[observer_name]
        plt.figure(figsize=(12, 6))
        plt.title(observer_name)
        for lst in list_of_lists:
            lst = [float(x) for x in lst]
            xs = np.arange(len(lst)).astype(float)
            xs += np.random.randn(*xs.shape) * 0.1
            plt.plot(xs, lst, "-", color="b", alpha=0.1, lw=1)
        plt.show()

    mean_log = histories[:, -1].mean()
    std_log = histories[:, -1].std()
    drops = histories.min(axis=1)
    drops_50 = np.quantile(drops, 0.50)
    drops_05 = np.quantile(drops, 0.05)
    drops_01 = np.quantile(drops, 0.01)
    stats = {
        "E[log(s_n / s_0)]": mean_log,
        "std[log(s_n / s_0)]": std_log,
        "F_drop(0.50)": drops_50,
        "F_drop(0.05)": drops_05,
        "F_drop(0.01)": drops_01,
        "num_episodes": num_episodes,
        "num_steps": histories.shape[1],
    }
    stats = {k: np.round(v, 3) for k, v in stats.items()}
    print("\nPolicy Evaluation Stats:")
    print("-" * 40)
    print(f"{'Metric':<20} | {'Value':>10}")
    print("-" * 40)
    for k, v in stats.items():
        print(f"{k:<20} | {v:>10}")
    print("-" * 40)
    assert abs(total_reward / num_episodes - mean_log) < 1e-9
    return (mean_log, stats, histories)


def get_ratio(state: StateVec, reward: float, total: float) -> float:
    adapter = StateAdapter(2, 4)
    cash = adapter.cash(state)
    stocks = adapter.stocks(state)
    prices = adapter.prices(state)
    in_stocks = (stocks * prices).sum()
    ratio = in_stocks / (in_stocks + cash)
    return ratio


if __name__ == "__main__":
    stocks, tech = get_data(
        ["GOOG", "AAPL"],
        "2023-01-01",
        "2024-01-01",
        "1D",
        {"RSI": rsi_14, "MACD": macd},
        ["rsi_14", "macd"],
    )
    seed = 42
    np.random.seed(seed)
    t.manual_seed(seed)
    env = StockTradingEnv(stocks, tech)
    policy = HODL(env.stock_dim, env.tech_dim)
    policy.ratio.data = t.tensor(0.99)
    policy.sigma.data = t.tensor(0.1)
    policy.k.data = t.tensor(2.0)
    mean_log, stats, histories = evaluate_policy(
        policy,
        env,
        num_episodes=100,
        observers={"ratio": get_ratio},
    )
