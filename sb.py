import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from data_processing import get_data, macd, rsi_14
from envs.debug import DebugEnv
from envs.stock_trading_env import StockTradingEnv
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv


# stocks, tech = get_data(
#     ["AAPL", "GOOGL"],
#     "2023-01-01",
#     "2024-01-01",
#     "1D",
#     {"RSI": rsi_14, "MACD": macd},
#     ["rsi_14", "macd"],
# )

# stocks_old, tech_old = get_data(
#     ["AAPL", "GOOGL"],
#     "2022-01-01",
#     "2023-01-01",
#     "1D",
#     {"RSI": rsi_14, "MACD": macd},
#     ["rsi_14", "macd"],
# )
# np.save("stocks.npy", stocks)
# np.save("tech.npy", tech)
# np.save("stocks_old.npy", stocks_old)
# np.save("tech_old.npy", tech_old)

stocks = np.load("stocks.npy")
tech = np.load("tech.npy")
stocks_old = np.load("stocks_old.npy")
tech_old = np.load("tech_old.npy")


def env_func(**kwargs):
    return StockTradingEnv(stocks, tech, **kwargs)


env_name = "stock"


def train():
    env = SubprocVecEnv([env_func for _ in range(4)])
    model = PPO("MlpPolicy", env, verbose=1)
    logger.info("Training model...")

    for _ in range(10):
        model.learn(total_timesteps=10**4)
        model.save(env_name)
        logger.info("Evaluating model...")

        env = StockTradingEnv(stocks, tech)
        mean_reward, std_reward = evaluate_policy(
            model,
            env,
            n_eval_episodes=10,
        )
        logger.info(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

        env = StockTradingEnv(stocks, tech)
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]
        mean_reward, std_reward = evaluate_policy(
            model,
            env,
            n_eval_episodes=10,
        )
        logger.info(
            f"Mean reward (dummy VecEnv): {mean_reward:.2f} +/- {std_reward:.2f}"
        )

        env = StockTradingEnv(stocks_old, tech_old)
        mean_reward, std_reward = evaluate_policy(
            model,
            env,
            n_eval_episodes=10,
        )
        logger.info(f"Mean reward (old data): {mean_reward:.2f} +/- {std_reward:.2f}")


def demo():
    # env = env_func()
    env = StockTradingEnv(stocks, tech)
    model = PPO.load(env_name)
    obs, _ = env.reset()
    env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=10,
        deterministic=True,
    )
    logger.info(f"Deterministic mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=10,
        deterministic=False,
    )
    logger.info(f"Stochastic mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    env = StockTradingEnv(stocks, tech)
    # env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]
    obs, _ = env.reset()
    cash_history = []
    wealth_history = []
    rewards = []
    n = stocks_old.shape[1]
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rewards.append(reward)
        cash = obs[0]
        p = obs[1 : n + 1]
        q = obs[n + 1 : 2 * n + 1]
        cash_history.append(cash)
        wealth = (q * p).sum() + cash
        wealth_history.append(wealth)
        if done:
            break
            obs, _ = env.reset()

    print(sum(rewards))
    # print(rewards[:10])
    # print(rewards[-10:])
    plt.plot(rewards, "g", label="rewards")
    plt.show()

    plt.plot(cash_history, "r", label="cash")
    plt.plot(wealth_history, "b", label="wealth")
    plt.legend()
    plt.yscale("log")
    plt.ylim(10**5, 10**7)
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
