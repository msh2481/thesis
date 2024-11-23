import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from envs.benchmark import CashRatioEnv, PredictableEnv
from loguru import logger
from models import fit_mlp_policy, MLPPolicy


def train_env(**kwargs):
    return PredictableEnv.create(n_stocks=1, tech_per_stock=1, n_steps=200)


def demo_env(**kwargs):
    return PredictableEnv.create(n_stocks=1, tech_per_stock=1, n_steps=200)


env_name = "predictable"


def train():
    env = train_env()
    model = fit_mlp_policy(env, n_epochs=1000, batch_size=4, lr=1e-9)


if __name__ == "__main__":
    train()
