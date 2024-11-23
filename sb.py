import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from data_processing import get_data, macd, rsi_14
from envs.benchmark import CashRatioEnv, PredictableEnv
from envs.debug import DebugEnv
from envs.stock_trading_env import StockTradingEnv
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

stocks_new = np.load("stocks_new.npy")
tech_new = np.load("tech_new.npy")
stocks_old = np.load("stocks_old.npy")
tech_old = np.load("tech_old.npy")
stocks_test = np.load("stocks_test.npy")
tech_test = np.load("tech_test.npy")


def train_env(**kwargs):
    return CashRatioEnv.create(n_stocks=2, tech_per_stock=1, n_steps=1000)


def demo_env(**kwargs):
    return CashRatioEnv.create(n_stocks=2, tech_per_stock=1, n_steps=1000)


env_name = "predictable"


def train():
    env = SubprocVecEnv([train_env for _ in range(4)])
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=1e-4)
    logger.info("Training model...")

    for _ in range(10):
        model.learn(total_timesteps=10**4)
        model.save(env_name)
        logger.info("Evaluating model...")

        env = train_env()
        mean_reward, std_reward = evaluate_policy(
            model,
            env,
            n_eval_episodes=3,
        )
        logger.info(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

        env = demo_env()
        mean_reward, std_reward = evaluate_policy(
            model,
            env,
            n_eval_episodes=3,
        )
        logger.info(f"Mean reward (validation): {mean_reward:.2f} +/- {std_reward:.2f}")


def demo():
    env = demo_env()
    model = PPO.load(env_name)
    obs, _ = env.reset()
    # env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=10,
        deterministic=True,
    )
    logger.info(f"Deterministic mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    obs, _ = env.reset()
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=10,
        deterministic=False,
    )
    logger.info(f"Stochastic mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    env = demo_env()
    # env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]
    obs, _ = env.reset()
    cash_history = []
    wealth_history = []
    rewards = []
    n = env.stock_dim
    q_last = None
    for iter in range(10**8):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rewards.append(reward)
        cash_ratio = obs[0]
        p = obs[1 : n + 1]
        q = obs[n + 1 : 2 * n + 1]
        wealth = env._get_log_total_asset(p)
        cash_history.append(cash_ratio)
        wealth_history.append(wealth)
        # if q_last is not None and (np.abs(q - q_last).sum() > 1e-9):
        #     print(iter, action, cash_ratio, p, q, wealth, env.log_total_asset(p))
        q_last = q
        if done:
            break
            obs, _ = env.reset()

    # print(sum(rewards))
    plt.subplot(2, 1, 1)
    plt.plot(cash_history, "r", label="cash")
    plt.subplot(2, 1, 2)
    W = np.array(wealth_history)
    plt.plot(np.log(W / W[0]), "k--", label="wealth")
    # X = stocks_old
    # for i in range(n):
    #     plt.plot(np.log(X[:, i] / X[0, i]), lw=1)
    plt.legend()
    plt.show()


def test():
    n = stocks_old.shape[1]

    def interact(env):
        if isinstance(env, VecEnv):
            obs = env.reset()
        else:
            obs, _ = env.reset()
        rewards = []
        cash_history = []
        for a in [-1.0, 1.0] * 3:
            action = np.array([a] * n)
            if isinstance(env, VecEnv):
                action = np.stack([action] * env.num_envs)
                obs, reward, done, info = env.step(action)
                cash = obs[0, 0]
            else:
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                cash = obs[0]
            rewards.append(reward)
            cash_history.append(cash)
            assert not done
        return rewards, cash_history

    for _ in range(3):
        env = StockTradingEnv(stocks_old, tech_old)
        rewards, cash_history = interact(env)
        print(rewards)
        print(cash_history)
        print("-----")
        env = StockTradingEnv(stocks_old, tech_old)
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]
        rewards, cash_history = interact(env)
        print(rewards)
        print(cash_history)
        print("=====")


def test2():
    def interact(model, env):
        if isinstance(env, VecEnv):
            obs = env.reset()
        else:
            obs, _ = env.reset()
        rewards = []
        cash_history = []
        for _ in range(100):
            action, _states = model.predict(obs, deterministic=True)
            if isinstance(env, VecEnv):
                obs, reward, done, info = env.step(action)
                reward = reward[0]
                cash = obs[0, 0]
            else:
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                cash = obs[0]
            rewards.append(reward)
            cash_history.append(cash)
            assert not done
        return rewards, cash_history

    model = PPO.load(env_name)
    for _ in range(3):
        env = StockTradingEnv(stocks_old, tech_old)
        rewards, cash_history = interact(model, env)
        print(sum(rewards))
        print(cash_history[-1])
        print("-----")
        env = StockTradingEnv(stocks_old, tech_old)
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]
        rewards, cash_history = interact(model, env)
        print(sum(rewards))
        print(cash_history[-1])
        print("=====")


if __name__ == "__main__":
    train()
    # demo()
    # test2()
