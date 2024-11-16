from elegantrl.agents import AgentModSAC
from elegantrl.train.config import Config, get_gym_env_args
from elegantrl.train.run import *
import gymnasium as gym

if __name__ == "__main__":
    env_func = gym.make
    env_args = {
        "env_num": 1,
        "env_name": "LunarLanderContinuous-v3",
        "max_step": 1000,
        "state_dim": 8,
        "action_dim": 2,
        "if_discrete": False,
        "target_return": 200,
    }
    args = Config(AgentModSAC, env_class=env_func, env_args=env_args)

    args.target_step = args.max_step
    args.gamma = 0.99
    args.eval_times = 2**5
    args.random_seed = 2022

    train_agent(args, if_single_process=True)
