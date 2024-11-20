import gymnasium as gym
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


def train():
    env = gym.make("LunarLanderContinuous-v3")
    model = PPO("MlpPolicy", env, verbose=1)
    logger.info("Training model...")
    model.learn(total_timesteps=100000)
    model.save("ppo_lunar")
    logger.info("Evaluating model...")
    mean_reward, std_reward = evaluate_policy(
        model,
        model.get_env(),
        n_eval_episodes=10,
    )
    logger.info(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


def demo():
    env = gym.make("LunarLanderContinuous-v3", render_mode="human")
    model = PPO.load("ppo_lunar")
    obs, _ = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()


if __name__ == "__main__":
    train()
    demo()
