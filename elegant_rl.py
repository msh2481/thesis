import copy
import glob
import os
import time
from pprint import pprint

import gymnasium as gym
import numpy as np
import torch as t
from beartype.typing import Callable
from elegantrl.agents import AgentA2C, AgentModSAC, AgentPPO
from elegantrl.train.config import Config, get_gym_env_args
from elegantrl.train.evaluator import Evaluator
from elegantrl.train.replay_buffer import ReplayBuffer
from elegantrl.train.run import train_agent, valid_agent
from envs.debug import DebugEnv

agent_cls = AgentA2C
# env_func = gym.make
# env_args = {
#     "env_num": 1,
#     "env_name": "LunarLanderContinuous-v3",
#     "max_step": 1000,
#     "state_dim": 8,
#     "action_dim": 2,
#     "if_discrete": False,
#     "target_return": 200,
#     "render_mode": "rgb_array",
# }
env_func = DebugEnv
env_args = {
    "env_num": 1,
    "env_name": "DebugEnv",
    "state_dim": 1,
    "action_dim": 1,
    "if_discrete": False,
    "target_return": 0.0,
}


def train():
    args = Config(agent_cls, env_class=env_func, env_args=env_args)
    args.target_step = args.max_step
    args.gamma = 0.99
    args.eval_times = 2**5
    args.eval_per_step = 2048
    args.random_seed = 2022
    args.reward_scale = 256
    args.learning_rate = 1e-5
    pprint(vars(args))

    train_agent(args, if_single_process=True)


def demo():
    env_func = gym.make
    env_args = copy.deepcopy(env_args)
    env_args["render_mode"] = "human"
    args = Config(agent_cls, env_class=env_func, env_args=env_args)
    args.target_step = args.max_step
    args.gamma = 0.99
    args.eval_times = 2**5
    args.eval_per_step = 2048
    args.random_seed = 2022
    pt_files = sorted(glob.glob("**/actor*.pt"))
    print("\nAvailable model files:")
    for i, file in enumerate(pt_files):
        print(f"{i}: {file}")
    choice = int(input("\nEnter the number of the model to load: "))
    actor_path = pt_files[choice]
    valid_agent(
        args.env_class,
        args.env_args,
        args.net_dims,
        agent_cls,
        actor_path,
    )


if __name__ == "__main__":
    train()
    # demo()
