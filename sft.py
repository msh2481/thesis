import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch as t
from envs.benchmark import (
    CashRatioEnv,
    MovingAverageEnv,
    PredictableEnv,
    TrendFollowingEnv,
)
from envs.diff_stock_trading_env import DiffStockTradingEnv
from loguru import logger
from models import fit_mlp_policy, imitation_rollout, MLPPolicy, rollout, TruePolicy
from tqdm import tqdm

stocks_new = t.tensor(np.load("stocks_new.npy"), dtype=t.float32)
tech_new = t.tensor(np.load("tech_new.npy"), dtype=t.float32)
stocks_old = t.tensor(np.load("stocks_old.npy"), dtype=t.float32)
tech_old = t.tensor(np.load("tech_old.npy"), dtype=t.float32)
stocks_test = t.tensor(np.load("stocks_test.npy"), dtype=t.float32)
tech_test = t.tensor(np.load("tech_test.npy"), dtype=t.float32)


def train_env(**kwargs):
    return TrendFollowingEnv.create(
        n_stocks=1, tech_per_stock=1, n_steps=1000, regenerate=False
    )


def val_env(**kwargs):
    return TrendFollowingEnv.create(
        n_stocks=1, tech_per_stock=1, n_steps=1000, regenerate=False
    )


def demo_env(**kwargs):
    return TrendFollowingEnv.create(
        n_stocks=1, tech_per_stock=1, n_steps=200, regenerate=False
    )


env_name = "predictable"


def train():
    model = fit_mlp_policy(
        env_factory=train_env,
        val_env_factory=val_env,
        val_period=5,
        n_epochs=1000,
        batch_size=1,
        lr=1e-4,
        rollout_fn=rollout,
        polyak_average=True,
        max_weight=10.0,
        langevin_coef=1e-6,
        # momentum=0.99,
        prior_std=1e9,
        dropout_rate=0.0,
        # init_from="checkpoints/good.pth",
    )


def demo(it: int, avg: bool):
    env = demo_env()
    steps = getattr(env, "n_steps", getattr(env, "max_step"))
    policy = MLPPolicy(env.state_dim, env.action_dim)
    if avg:
        policy.load_state_dict(
            t.load(f"checkpoints/avg_policy_{it}.pth", weights_only=False)
        )
    else:
        policy.load_state_dict(
            t.load(f"checkpoints/policy_{it}.pth", weights_only=False)
        )
    policy.eval()
    state, _ = env.reset_t()

    # Lists to store trajectory
    cash_ratio_history = []  # scalar
    cash_history = []  # scalar
    action_history = []  # [n_stocks]
    price_history = []  # [n_stocks]
    stock_history = []  # [n_stocks]
    tech_history = []  # [n_stocks]
    reward_history = []  # scalar
    portfolio_value_history = []  # scalar
    initial_cash = env.cash.item()
    n_stocks = env.stock_dim

    # Destructure initial state
    cash_ratio = state[0]  # scalar
    prices = state[1 : 1 + n_stocks]  # n_stocks
    stocks = state[1 + n_stocks : 1 + 2 * n_stocks]  # n_stocks
    techs = state[1 + 2 * n_stocks : 1 + 3 * n_stocks]  # n_stocks
    cash = state[1 + 3 * n_stocks]  # scalar
    cash_div_prices = state[1 + 3 * n_stocks + 1 :]  # n_stocks

    for _ in range(steps):
        action = policy.predict(state, deterministic=True)
        state, reward, terminated, truncated, info = env.step_t(action)

        # Destructure new state
        cash_ratio = state[0]
        prices = state[1 : 1 + n_stocks]
        stocks = state[1 + n_stocks : 1 + 2 * n_stocks]
        techs = state[1 + 2 * n_stocks : 1 + 3 * n_stocks]
        cash = state[1 + 3 * n_stocks]
        cash_div_prices = state[1 + 3 * n_stocks + 1 :]

        penalty = env._get_penalty()
        if penalty > 0.0:
            logger.warning(f"Penalty: {penalty.item()}")

        # Store vectors and scalars
        cash_ratio_history.append(cash_ratio.item())
        cash_history.append(cash.item())
        action_history.append(action.detach().cpu().numpy())
        price_history.append(prices.detach().cpu().numpy())
        stock_history.append(stocks.detach().cpu().numpy())
        tech_history.append(techs.detach().cpu().numpy())
        reward_history.append(reward.item())
        portfolio_value_history.append(cash.item() + (stocks * prices).sum().item())

        done = terminated or truncated
        if done:
            break

    # Convert lists to arrays
    cash_ratio_history = np.array(cash_ratio_history)
    cash_history = np.array(cash_history)
    action_history = np.array(action_history)
    price_history = np.array(price_history)
    stock_history = np.array(stock_history)
    tech_history = np.array(tech_history)
    portfolio_value_history = np.array(portfolio_value_history)

    # Calculate buy & hold returns based on average cash ratio
    avg_cash_ratio = cash_ratio_history.mean()
    buy_hold_values = []

    # Initial allocation based on average cash ratio
    initial_stock_value = (1 - avg_cash_ratio) * initial_cash
    initial_stock_units = (
        initial_stock_value / price_history[0].sum()
    )  # Equally weighted portfolio
    initial_cash_held = avg_cash_ratio * initial_cash

    for price in price_history:  # price is [n_stocks]
        # For each timestep, calculate portfolio value with constant allocation
        stock_value = initial_stock_units * price.sum()
        value = initial_cash_held + stock_value
        buy_hold_values.append(value)
    buy_hold_values = np.array(buy_hold_values)

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 3, 1)
    plt.plot(cash_ratio_history, "b-", label="Cash Ratio")
    plt.axhline(avg_cash_ratio, color="k", linestyle="--", label="Avg Cash Ratio")
    plt.axhline(0, color="k", linestyle=":")
    plt.legend()

    plt.subplot(2, 3, 2)
    for i in range(n_stocks):
        plt.plot(price_history[:, i], f"C{i}-", label=f"Price {i}")
        plt.plot(tech_history[:, i], f"C{i}--", label=f"Tech {i}")
    plt.axhline(0, color="k", linestyle="--")
    plt.legend()

    plt.subplot(2, 3, 3)
    for i in range(n_stocks):
        plt.plot(stock_history[:, i], f"C{i}-", label=f"Stock {i}")
    plt.axhline(0, color="k", linestyle=":")
    plt.legend()

    plt.subplot(2, 3, 4)
    for i in range(n_stocks):
        plt.plot(action_history[:, i], f"C{i}-", label=f"Action {i}")
    plt.axhline(0, color="k", linestyle="--")
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(np.cumsum(reward_history), "r-", label="Strategy Cum. Reward")
    strategy_value_change = portfolio_value_history - portfolio_value_history[0]
    buyhold_value_change = np.log(buy_hold_values) - np.log(buy_hold_values[0])
    plt.plot(buyhold_value_change, "b--", label="Buy & Hold Cum. Return")
    plt.axhline(0, color="k", linestyle="--")
    plt.legend()

    plt.subplot(2, 3, 6)
    for i in range(n_stocks):
        plt.plot(price_history[:, i], f"C{i}-", label=f"Price {i}")
        plt.plot(action_history[:, i], f"C{i}--", alpha=0.5, label=f"Action {i}")
    plt.axhline(0, color="k", linestyle="--")
    plt.legend()

    plt.tight_layout()
    plt.savefig("logs/demo.png")
    plt.close()


def evaluate():
    n = 50
    results = []
    for _ in tqdm(range(n)):
        env = demo_env()
        policy = TruePolicy(env.state_dim, env.action_dim)
        result = rollout(policy, env, deterministic=True)
        results.append(result.item())
    print("Median reward:", np.median(results))
    print("Min reward:", np.min(results))
    print("Max reward:", np.max(results))
    print(f"Average reward: {np.mean(results)}")
    print(f"Std: {np.std(results) / np.sqrt(n)}")


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select mode and parameters")
    parser.add_argument(
        "mode",
        choices=["train", "demo"],
        help="Mode to run: train or demo",
    )
    parser.add_argument(
        "--iter", type=int, default=100, help="Number of iterations to load for demo"
    )
    parser.add_argument("--avg", action="store_true", help="Use Polyak-averaged policy")
    args = parser.parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "demo":
        demo(args.iter, args.avg)
