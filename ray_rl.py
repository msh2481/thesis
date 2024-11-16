from collections import defaultdict
from concurrent.futures import as_completed, ThreadPoolExecutor
from copy import deepcopy
from pprint import pprint

import gymnasium as gym
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
from beartype import beartype as typed
from jaxtyping import Float, Int
from loguru import logger
from numpy import ndarray as ND
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import JsonLoggerCallback, NoopLogger
from torch import nn, Tensor as TT
from torch.distributions import Distribution
from torch.nn import functional as F
from tqdm.auto import tqdm

if __name__ == "__main__":
    config = PPOConfig()
    config.api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True,
    )
    config.framework("torch")
    config.environment("MountainCar-v0", env_config={"max_episode_steps": 4000})
    config.env_runners(num_env_runners=1)
    config.training(gamma=0.9, lr=0.01, kl_coeff=0.3, train_batch_size_per_learner=256)
    algo = config.build(logger_creator=lambda config: NoopLogger(config, ""))
    algo.train()

    for i in range(10):
        result = algo.train()
        result.pop("config")
        pprint(result["env_runners"])
