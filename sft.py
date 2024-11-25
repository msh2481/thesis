import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch as t
from envs.benchmark import CashRatioEnv, PredictableEnv
from loguru import logger
from models import fit_mlp_policy, imitation_rollout, MLPPolicy, TruePolicy


def train_env(**kwargs):
    return PredictableEnv.create(n_stocks=1, tech_per_stock=1, n_steps=200)


def demo_env(**kwargs):
    return PredictableEnv.create(n_stocks=1, tech_per_stock=1, n_steps=200)


env_name = "predictable"


def train():
    env = train_env()
    model = fit_mlp_policy(
        env,
        n_epochs=1000,
        batch_size=1,
        lr=1e-3,
        # rollout_fn=imitation_rollout,
        # init_from="checkpoints/good.pth",
    )


def demo():
    it = 980
    steps = 20

    env = demo_env()
    policy = MLPPolicy(env.state_dim, env.action_dim)
    policy.load_state_dict(t.load(f"checkpoints/policy_{it}.pth", weights_only=False))
    # policy = TruePolicy(env.state_dim, env.action_dim)
    state, _ = env.reset_t()
    cashes = []
    actions = []
    prices = []
    stocks = []
    techs = []
    rewards = []

    for _ in range(steps):
        cash_ratio, price, stock, tech = state
        cash = env.cash
        action = policy.predict(state, deterministic=True)
        state, reward, terminated, truncated, info = env.step_t(action)

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
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(prices, "r-", label="Price")
    plt.plot(techs, "m-", label="Technical Signal")
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.plot(stocks, "g-", label="Stock Holdings")
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

    plt.tight_layout()
    plt.savefig("logs/demo.png")
    plt.close()


if __name__ == "__main__":
    # train()
    demo()
