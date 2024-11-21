from collections import defaultdict
from concurrent.futures import as_completed, ThreadPoolExecutor
from copy import deepcopy

import gymnasium as gym
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
from beartype import beartype as typed
from beartype.typing import Callable, Iterable, Iterator
from data_processing import get_data, macd, rsi_14
from envs.stock_trading_env import ActionVec, StateVec, StockTradingEnv
from gymnasium import Env
from jaxtyping import Float, Int
from loguru import logger
from models import Actor, fit_actor, HODL, MountainCarPolicy, StateAdapter
from numpy import ndarray as ND
from torch import nn, Tensor as TT
from torch.distributions import Distribution
from torch.nn import functional as F
from tqdm.auto import tqdm


@typed
def run_policy(
    policy: Actor,
    env: Env,
    save_video_to: str | None = None,
    observers: dict[str, Callable[[StateVec, float, float], float]] = {},
) -> tuple[list[tuple], list[float], list[float], dict[str, list[float]]]:
    states = []
    terminated = False
    truncated = False
    state, _ = env.reset()
    state_action_pairs = []
    rewards = []
    totals = []
    frames = []
    metrics = defaultdict(list)
    total = 0
    while not terminated and not truncated:
        if isinstance(state, ND):
            state = t.tensor(state)
        action = policy.actor(state).sample()
        next_state, reward, terminated, truncated, _ = env.step(action.numpy())
        state_action_pairs.append((state, action))
        rewards.append(reward)
        total += reward
        totals.append(total)
        state = next_state
        states.append(state)
        for observer_name, observer in observers.items():
            metrics[observer_name].append(observer(state, reward, total))
        if save_video_to is not None:
            frames.append(env.render())
    if save_video_to is not None:
        imageio.mimsave(save_video_to, frames, fps=30)
    return state_action_pairs, rewards, totals, metrics


@typed
def demonstrate_policy(
    policy: Actor,
    env: Env,
    num_episodes: int = 100,
    observers: dict[str, Callable[[StateVec, float, float], float]] = {},
) -> tuple[float, dict[str, float], Float[ND, "num_episodes num_steps"]]:
    total_reward = 0
    all_totals = []
    all_metrics = defaultdict(list)
    for _ in tqdm(range(num_episodes), desc="Evaluating policy"):
        states, rewards, totals, metrics = run_policy(policy, env, observers=observers)
        total_reward += totals[-1]
        all_totals.append(totals)
        for observer_name in observers:
            all_metrics[observer_name].append(metrics[observer_name])

    max_len = max(len(seq) for seq in all_totals)
    padded_totals = []
    for seq in all_totals:
        if len(seq) < max_len:
            padded_seq = seq + [seq[-1]] * (max_len - len(seq))
        else:
            padded_seq = seq
        padded_totals.append(padded_seq)
    histories = np.array(padded_totals)

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


@typed
def show_policy(policy: Actor, env: Env, filename: str, fps: int = 30):
    state, _ = env.reset()
    terminated = False
    truncated = False
    frames = []
    while not terminated and not truncated:
        if isinstance(state, ND):
            state = t.tensor(state)
        action = policy.actor(state).sample().numpy()
        state, _, terminated, truncated, _ = env.step(action)
        frames.append(env.render())
    logger.info(f"Len = {len(frames)}")
    if filename is not None:
        imageio.mimsave(filename, frames, fps=fps)


@typed
def cross_entropy_method(
    create_env: Callable[[], Env],
    policy: Actor,
    n_epochs: int = 10,
    n_samples: int = 100,
    buffer_size: int = 100,
    show_policies: bool = True,
    ratio: float = 0.2,
    learning_rate: float = 0.01,
    gradient_steps: int = 10,
) -> Actor:
    buffer = []

    for epoch in range(n_epochs):
        logger.info(f"Epoch {epoch}")

        # Collect trajectories
        returns = []
        trajectories = []
        if parallel := True:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for _ in range(n_samples):

                    def collect_trajectory():
                        env = create_env()
                        state_action_pairs, _, totals, _ = run_policy(policy, env)
                        return totals[-1], state_action_pairs

                    futures.append(executor.submit(collect_trajectory))

                for future in tqdm(as_completed(futures), total=len(futures)):
                    total_reward, trajectory = future.result()
                    returns.append(total_reward)
                    trajectories.append(trajectory)
                    buffer.append((total_reward, trajectory))
        else:
            for _ in tqdm(range(n_samples), desc="Collecting trajectories"):
                env = create_env()
                state_action_pairs, rewards, totals, _ = run_policy(policy, env)
                total_reward = sum(rewards)
                returns.append(total_reward)
                trajectories.append(state_action_pairs)
                buffer.append((total_reward, state_action_pairs))

        # Keep only best trajectories
        buffer = buffer[-buffer_size:]
        good_buffer = sorted(buffer, key=lambda x: x[0], reverse=True)
        k = max(1, int(len(good_buffer) * ratio))
        mean_return = np.mean([x[0] for x in good_buffer[:k]])
        elite_trajectories = [x[1] for x in good_buffer[:k]]
        logger.info(f"Mean return of elite trajectories: {mean_return:.2f}")

        # Update policy to maximize likelihood of elite trajectories
        elite_dataset = []
        for trajectory in elite_trajectories:
            for state, action in trajectory:
                if isinstance(state, ND):
                    state = t.tensor(state)
                if isinstance(action, ND):
                    action = t.tensor(action)
                elite_dataset.append((state, action))
        fit_actor(
            policy,
            elite_dataset,
            lr=learning_rate,
            n_epochs=gradient_steps,
            batch_size=buffer_size,
        )

        returns = np.array(returns)
        quantiles = np.quantile(returns, [0.01, 0.25, 0.5, 0.75, 0.99])
        logger.info(
            f"Return quantiles: [{' | '.join(f'{q:.2f}' for q in quantiles)}]",
        )

        if show_policies:
            env = create_env()
            show_policy(policy, env, f"logs/policy_{epoch}.mp4")

    return policy


def test_evaluation():
    sp500_table = pd.read_html(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    )
    sp500_tickers = sp500_table[0]["Symbol"].tolist()
    stocks, tech = get_data(
        sp500_tickers,
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

    def get_ratio(state: StateVec, reward: float, total: float) -> float:
        adapter = StateAdapter(env.stock_dim, env.tech_dim)
        cash = adapter.cash(state)
        stocks = adapter.stocks(state)
        prices = adapter.prices(state)
        in_stocks = (stocks * prices).sum()
        ratio = in_stocks / (in_stocks + cash)
        return ratio

    mean_log, stats, histories = demonstrate_policy(
        policy,
        env,
        num_episodes=10,
        observers={"profit": lambda s, r, t: t, "ratio": get_ratio},
    )


def test_cem():
    policy = MountainCarPolicy()
    policy, _ = cross_entropy_method(
        lambda: gym.make(
            "MountainCar-v0", render_mode="rgb_array", max_episode_steps=10000
        ),
        policy,
        n_epochs=100,
        n_samples=10,
        buffer_size=200,
        show_policies=True,
        learning_rate=1e-3,
        gradient_steps=10,
    )


if __name__ == "__main__":
    test_evaluation()
