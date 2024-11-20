import os
import time

import gymnasium as gym
import numpy as np
import torch as t
from beartype.typing import Callable
from elegantrl.agents import AgentModSAC
from elegantrl.train.config import Config, get_gym_env_args
from elegantrl.train.evaluator import Evaluator
from elegantrl.train.replay_buffer import ReplayBuffer
from elegantrl.train.run import train_agent, valid_agent


def train():
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
    args.eval_per_step = 2048
    args.random_seed = 2022

    train_agent(args, if_single_process=True)


def demo():
    env_func = gym.make
    env_args = {
        "env_num": 1,
        "env_name": "LunarLanderContinuous-v3",
        "max_step": 1000,
        "state_dim": 8,
        "action_dim": 2,
        "if_discrete": False,
        "target_return": 200,
        "render_mode": "human",
    }
    args = Config(AgentModSAC, env_class=env_func, env_args=env_args)
    args.target_step = args.max_step
    args.gamma = 0.99
    args.eval_times = 2**5
    args.eval_per_step = 2048
    args.random_seed = 2022
    actor_path = "LunarLanderContinuous-v3_ModSAC_2022/actor__000000080384.pt"
    valid_agent(
        args.env_class,
        args.env_args,
        args.net_dims,
        AgentModSAC,
        actor_path,
    )


if __name__ == "__main__":
    # train()
    demo()
