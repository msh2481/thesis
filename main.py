import matplotlib.pyplot as plt
import numpy as np
import torch as t
from envs.benchmark import gen_ma, gen_trend, make_ema_tech
from envs.diff_stock_trading_env import DiffStockTradingEnv
from loguru import logger
from models import fit_policy, MLPPolicy, rollout
from tqdm import tqdm

# stocks_new = t.tensor(np.load("stocks_new.npy"), dtype=t.float32)
# tech_new = t.tensor(np.load("tech_new.npy"), dtype=t.float32)
# stocks_old = t.tensor(np.load("stocks_old.npy"), dtype=t.float32)
# tech_old = t.tensor(np.load("tech_old.npy"), dtype=t.float32)
# stocks_test = t.tensor(np.load("stocks_test.npy"), dtype=t.float32)
# tech_test = t.tensor(np.load("tech_test.npy"), dtype=t.float32)

fn = gen_trend
full_train_prices = fn(10**5, 1)
full_train_tech = make_ema_tech(full_train_prices)
full_train_env = DiffStockTradingEnv(full_train_prices, full_train_tech)
full_val_prices = fn(10**5, 1)
full_val_tech = make_ema_tech(full_val_prices)
full_val_env = DiffStockTradingEnv(full_val_prices, full_val_tech)


def train_env(**kwargs):
    train_length = 1000
    l = np.random.randint(0, full_train_env.price_array.shape[0] - train_length)
    r = l + train_length
    return full_train_env.subsegment(l, r)


def val_env(**kwargs):
    val_length = 1000
    l = np.random.randint(0, full_val_env.price_array.shape[0] - val_length)
    r = l + val_length
    return full_val_env.subsegment(l, r)


def demo_env(**kwargs):
    demo_length = 1000
    l = np.random.randint(0, full_val_env.price_array.shape[0] - demo_length)
    r = l + demo_length
    return full_val_env.subsegment(l, r)


def train():
    model = fit_policy(
        env_factory=train_env,
        val_env_factory=val_env,
        val_period=5,
        n_epochs=1000,
        batch_size=1,
        lr=1e-3,
        rollout_fn=rollout,
        polyak_average=True,
        max_weight=10.0,
        langevin_coef=0.0,  # 1e-4,
        # prior_std=10.0,
        dropout_rate=0.0,
        # init_from="checkpoints/tf.pth",
    )


def demo(it: int, avg: bool):
    env = demo_env()
    steps = env.n_steps
    policy = MLPPolicy(
        env.observation_space.shape,
        env.action_space.shape[0],
        dropout_rate=0.9,
    )
    if avg:
        policy.load_state_dict(
            t.load(f"checkpoints/avg_policy_{it}.pth", weights_only=False)
        )
    else:
        policy.load_state_dict(
            t.load(f"checkpoints/policy_{it}.pth", weights_only=False)
        )
    policy.eval()
    state, _ = env.reset_state()

    # Lists to store trajectory
    cash_history = []
    position_history = []
    price_history = []
    tech_history = []
    reward_history = []
    value_history = []
    initial_value = state.value().item()

    with t.no_grad():
        for _ in range(steps):
            action = policy.predict(
                state.features(env.numpy, env.flat), noise_level=0.0
            )
            state, reward, terminated, truncated, _ = env.make_step(action)
            # Store state information
            cash_history.append(state.cash.item())
            position_history.append(state.position().numpy())
            price_history.append(state.prices.numpy())
            tech_history.append(state.tech.numpy())
            reward_history.append(reward.item())
            value_history.append(state.value().item())

            done = terminated or truncated
            if done:
                break

    # Convert lists to arrays
    cash_history = np.array(cash_history)
    position_history = np.array(position_history)
    price_history = np.array(price_history)
    tech_history = np.array(tech_history)
    value_history = np.array(value_history)
    logreturn_history = np.log(value_history / value_history[0])

    # Calculate buy & hold returns
    avg_pos = np.mean(position_history, axis=0)
    logger.debug(f"Average position: {avg_pos}")
    buy_hold_logreturns = []
    buy_hold_logreturn = 0
    last_prices = price_history[0]
    for prices in price_history:
        stocks = avg_pos / last_prices
        cash = 1 - avg_pos.sum()
        current_return = ((prices * stocks).sum() + cash) / (
            (last_prices * stocks).sum() + cash
        )
        buy_hold_logreturn += np.log(current_return)
        buy_hold_logreturns.append(buy_hold_logreturn)
        last_prices = prices
    buy_hold_logreturns = np.array(buy_hold_logreturns)

    # Plot results
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.title("Portfolio Value")
    plt.plot(logreturn_history, "b-", label="Policy")
    plt.plot(buy_hold_logreturns, "r--", label="Average position")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.title("Asset Prices")
    for i in range(env.stock_dim):
        plt.plot(price_history[:, i], label=f"Asset {i+1}")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.title("Portfolio Positions")
    for i in range(env.stock_dim):
        plt.plot(position_history[:, i], label=f"Asset {i+1}")
    plt.plot(1 - position_history.sum(axis=1), label="Cash")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.title("Cumulative Returns")
    plt.plot(np.cumsum(reward_history), "g-")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("logs/demo.png")
    plt.close()


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
