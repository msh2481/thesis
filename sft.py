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
from loguru import logger
from models import fit_mlp_policy, imitation_rollout, MLPPolicy, rollout, TruePolicy
from tqdm import tqdm


def train_env(**kwargs):
    return TrendFollowingEnv.create(n_stocks=1, tech_per_stock=1, n_steps=200)


def demo_env(**kwargs):
    return TrendFollowingEnv.create(n_stocks=1, tech_per_stock=1, n_steps=200)


env_name = "predictable"


def train():
    env = train_env()
    model = fit_mlp_policy(
        env,
        n_epochs=1000,
        batch_size=1,
        lr=1e-3,
        rollout_fn=rollout,
        # init_from="checkpoints/good.pth",
    )


def demo():
    it = 400
    steps = 80

    env = demo_env()
    policy = MLPPolicy(env.state_dim, env.action_dim)
    policy.load_state_dict(t.load(f"checkpoints/policy_{it}.pth", weights_only=False))
    # policy.load_state_dict(t.load(f"checkpoints/good.pth", weights_only=False))
    # policy = TruePolicy(env.state_dim, env.action_dim)
    state, _ = env.reset_t()
    cashes = []
    actions = []
    prices = []
    stocks = []
    techs = []
    rewards = []

    for _ in range(steps):
        cash_ratio, price, stock, tech, cash, cash_div_price = state
        cash = env.cash
        action = policy.predict(state, deterministic=True)
        state, reward, terminated, truncated, info = env.step_t(action)
        penalty = env._get_penalty()
        if penalty > 0.0:
            logger.warning(f"Penalty: {penalty.item()}")

        cashes.append(cash.item())
        actions.append(action.item())
        prices.append(price.item())
        stocks.append(stock.item())
        techs.append(tech.item())
        rewards.append(reward.item())

        done = terminated or truncated
        if done:
            break

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 3, 1)
    plt.plot(cashes, "b-", label="Cash")
    plt.axhline(0, color="k", linestyle="--")
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(prices, "r-", label="Price")
    plt.plot(techs, "m-", label="Technical Signal")
    plt.axhline(0, color="k", linestyle="--")
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.plot(stocks, "g-", label="Stock Holdings")
    plt.axhline(0, color="k", linestyle="--")
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

    plt.subplot(2, 3, 6)
    plt.plot(prices, "r-", label="Price")
    plt.plot(actions, "m-", label="Action")
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


if __name__ == "__main__":
    # train()
    demo()
    # evaluate()
