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
from envs.stock_trading_env import StateVec, StockTradingEnv
from gymnasium import Env
from jaxtyping import Float, Int
from loguru import logger
from models import Actor, HODL, StateAdapter
from numpy import ndarray as ND
from torch import nn, Tensor as TT
from torch.distributions import Distribution
from torch.nn import functional as F
from tqdm.auto import tqdm


@typed
def evaluate_policy(
    policy: Actor,
    env: Env,
    num_episodes: int = 100,
    observers: dict[str, Callable[[StateVec, float, float], float]] = {},
    extra_info: bool = False,
) -> tuple[float, dict[str, float], Float[ND, "num_episodes num_steps"]] | float:
    total_reward = 0
    all_totals = []
    all_metrics = defaultdict(list)
    episode_iterable = (
        tqdm(range(num_episodes), desc="Evaluating policy")
        if extra_info
        else range(num_episodes)
    )
    for _ in episode_iterable:
        current_totals = []
        current_total = 0
        current_metrics = defaultdict(list)
        state, _ = env.reset()
        terminated = False
        truncated = False
        while not terminated and not truncated:
            if isinstance(state, ND):
                state = t.tensor(state)
            action = policy.actor(state).sample().numpy()
            next_state, reward, terminated, truncated, _ = env.step(action)
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

    if not extra_info:
        return total_reward / num_episodes

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
def flatten_model_parameters(model: nn.Module) -> Float[TT, "num_params"]:
    return t.cat([p.view(-1) for p in model.parameters() if p.requires_grad])


@typed
def unflatten_model_parameters(
    model: nn.Module,
    flat_params: Float[TT, "num_params"],
) -> nn.Module:
    idx = 0
    model = deepcopy(model)
    for p in model.parameters():
        num_params = p.numel()
        p.data = flat_params[idx : idx + num_params].view(p.shape)
        idx += num_params
    return model


@typed
def estimate_gaussian(
    samples: Float[TT, "num_samples dim"],
) -> Distribution:
    n, d = samples.shape
    mean = t.zeros(d)
    cov = t.eye(d)
    if n > 0:
        mean = samples.mean(dim=0)
    if n > 1:
        cov = ((samples - mean).T @ (samples - mean) + t.eye(d)) / (n - 1)
    return t.distributions.MultivariateNormal(loc=mean, covariance_matrix=cov)


@typed
def cross_entropy_method(
    create_env: Callable[[], Env],
    policy: Actor,
    n_epochs: int = 10,
    n_samples: int = 100,
    buffer_size: int = 100,
    show_policies: bool = True,
    ratio: float = 0.2,
) -> list[Actor]:
    w0 = flatten_model_parameters(policy)
    buffer = [(-float("inf"), w0)]
    model_qualities = []
    good_params = []

    for epoch in range(n_epochs):
        logger.info(f"Epoch {epoch}")
        buffer = buffer[-buffer_size:]
        good_buffer = list(buffer)
        good_buffer.sort(key=lambda x: x[0], reverse=True)
        k = max(1, int(len(good_buffer) * ratio))
        model_qualities = [x[0] for x in good_buffer[:k]]
        logger.info(f"Mean good quality: {np.mean(model_qualities):.2f}")
        good_params = [x[1] for x in good_buffer[:k]]
        dist = estimate_gaussian(t.stack(good_params))
        samples = dist.sample((n_samples,))

        returns = []
        if parallel := True:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for sample in samples:

                    def evaluate_sample(sample):
                        env = create_env()
                        unflatten_model_parameters(policy, sample)
                        mean_return = evaluate_policy(policy, env, num_episodes=1)
                        return mean_return, sample

                    futures.append(executor.submit(evaluate_sample, sample))

                for future in tqdm(as_completed(futures), total=len(futures)):
                    mean_return, sample = future.result()
                    returns.append(mean_return)
                    buffer.append((mean_return, sample))
        else:
            for sample in tqdm(samples, desc="Evaluating samples"):
                unflatten_model_parameters(policy, sample)
                mean_return, _, _ = evaluate_policy(policy, env, num_episodes=10)
                returns.append(mean_return)
                good_buffer.append((mean_return, sample))

        returns = np.array(returns)
        quantiles = np.quantile(returns, [0.01, 0.25, 0.5, 0.75, 0.99])
        logger.info(
            f"Return quantiles: [{' | '.join(f'{q:.2f}' for q in quantiles)}]",
        )

        if show_policies:
            best_policy = unflatten_model_parameters(policy, good_params[0])
            env = create_env()
            show_policy(best_policy, env, f"logs/policy_{epoch}.mp4")

    logger.info("Model qualities on last iteration:")
    for i, m in enumerate(model_qualities):
        logger.info(f"{i:>2} | {m:.2f}")
    models = [unflatten_model_parameters(policy, x) for x in good_params]
    return models, model_qualities


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

    mean_log, stats, histories = evaluate_policy(
        policy,
        env,
        num_episodes=10,
        observers={"profit": lambda s, r, t: t, "ratio": get_ratio},
    )


def test_gaussian_fit():
    x = np.random.randn(100, 1)
    y = np.random.randn(100, 1)
    points = np.concatenate([x + 2 * y, y + 2 * x], axis=1)
    points = t.tensor(points, dtype=t.float32)
    points = t.zeros((0, 2))
    dist = estimate_gaussian(points)
    more_samples = dist.sample((1000,))
    plt.scatter(points[:, 0], points[:, 1], label="samples")
    plt.scatter(more_samples[:, 0], more_samples[:, 1], label="samples", alpha=0.1)
    plt.legend()
    plt.show()


class MountainCarPolicy(Actor):
    @typed
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 3)

    @typed
    def actor(self, x: Float[TT, "2"]):
        x = self.fc(x)
        x = F.relu(x)
        x = self.fc2(x)
        return t.distributions.Categorical(logits=x)

    @typed
    def actor_parameters(self) -> Iterable[nn.Parameter]:
        return self.parameters()


def test_cem():
    policy = MountainCarPolicy()
    models, model_qualities = cross_entropy_method(
        lambda: gym.make(
            "MountainCar-v0", render_mode="rgb_array", max_episode_steps=10000
        ),
        policy,
        n_epochs=20,
        n_samples=100,
        buffer_size=200,
        show_policies=True,
    )


if __name__ == "__main__":
    test_cem()
