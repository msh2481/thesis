import numpy as np
import torch as t
from beartype import beartype as typed
from beartype.typing import Iterable

from envs.stock_trading_env import ActionVec, StateVec
from jaxtyping import Float, Int
from numpy import ndarray as ND
from torch import nn, Tensor as TT
from torch.distributions import Distribution
from torch.nn import functional as F
from tqdm.auto import tqdm

"""
First I want a simple model that just keeps a certain ratio of money in stocks, uniformly. 
This ratio should be the (only) trainable parameter.

I also want to have something like actor-critic in terms of algorithms.
For that the model should have critic(s, a) -> Q-value (as a scalar or distribution) and actor(s) -> distribution over actions.
Then, depending on the properties of this distribution and of the critic, I can either tune its density pointwise (like in A2C), or tune reparametrized samples (like in DDPG).

And also I want another type of algorithm, Decision G.pt: 
It will sample and then learn mapping from rewards to model weights :)
"""

from abc import ABC, abstractmethod


class Actor(nn.Module, ABC):
    """Base class for actor models that output action distributions"""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def actor(self, state) -> Distribution:
        raise NotImplementedError()

    @typed
    def actions_are_differentiable(self) -> bool:
        return False

    @abstractmethod
    def actor_parameters(self) -> Iterable[t.nn.Parameter]:
        raise NotImplementedError()


class StateAdapter:
    def __init__(self, stock_dim: int, tech_dim: int):
        self.stock_dim = stock_dim
        self.tech_dim = tech_dim
        self.state_dim = 2 + 3 * stock_dim + tech_dim
        """
        cash_scaled,
        price_scaled,
        stock_scaled,
        self.stocks_cool_down,
        self.tech_array[self.time],
        max_stock,
        """

    def ith_stock(self, state: StateVec, i: int) -> Float[TT, "4"]:
        assert state.shape == (self.state_dim,)
        """Returns cash, price[i], stock[i], cooldown[i]"""
        n = self.stock_dim
        return t.tensor(
            [
                state[0],
                state[1 + i],
                state[1 + n + i],
                state[1 + 2 * n + i],
            ],
            dtype=t.float32,
        )

    def cash(self, state: StateVec) -> Float[TT, ""]:
        assert state.shape == (self.state_dim,)
        return t.tensor(state[0], dtype=t.float32)

    def max_stock(self, state: StateVec) -> Float[TT, ""]:
        assert state.shape == (self.state_dim,)
        return t.tensor(state[-1], dtype=t.float32)

    def prices(self, state: StateVec) -> Float[TT, "n"]:
        assert state.shape == (self.state_dim,)
        return t.tensor(state[1 : 1 + self.stock_dim], dtype=t.float32)

    def stocks(self, state: StateVec) -> Float[TT, "n"]:
        assert state.shape == (self.state_dim,)
        return t.tensor(
            state[1 + self.stock_dim : 1 + 2 * self.stock_dim], dtype=t.float32
        )

    def cooldowns(self, state: StateVec) -> Float[TT, "n"]:
        assert state.shape == (self.state_dim,)
        return t.tensor(
            state[1 + 2 * self.stock_dim : 1 + 3 * self.stock_dim], dtype=t.float32
        )

    def tech(self, state: StateVec) -> Float[TT, "tech_dim"]:
        assert state.shape == (self.state_dim,)
        return t.tensor(
            state[1 + 3 * self.stock_dim : 1 + 3 * self.stock_dim + self.tech_dim],
            dtype=t.float32,
        )


class HODL(Actor):
    """Model that keeps a constant ratio of money in stocks"""

    @typed
    def __init__(self, stock_dim: int, tech_dim: int):
        super().__init__()
        self.ratio = t.nn.Parameter(t.tensor(0.5))
        self.sigma = t.nn.Buffer(t.tensor(0.1))
        self.k = t.nn.Buffer(t.tensor(0.1))
        self.stock_dim = stock_dim
        self.tech_dim = tech_dim
        self.state_adapter = StateAdapter(stock_dim, tech_dim)

    @typed
    def actor(self, state: StateVec) -> Distribution:
        cash = self.state_adapter.cash(state)
        in_stocks = (
            self.state_adapter.stocks(state) * self.state_adapter.prices(state)
        ).sum()
        current_ratio = in_stocks / (cash + in_stocks)
        mu = t.ones(self.stock_dim) * (self.ratio - current_ratio) * self.k
        sigma = t.eye(self.stock_dim) * self.sigma
        return t.distributions.MultivariateNormal(loc=mu, scale_tril=sigma)

    @typed
    def actions_are_differentiable(self) -> bool:
        return True

    @typed
    def actor_parameters(self) -> Iterable[t.nn.Parameter]:
        return [self.ratio]


# TODO: take log of prices and cash before feeding into the model


@typed
def fit_actor(
    actor: Actor,
    dataset: list[tuple[TT, TT]],
    lr: float,
    n_epochs: int,
    batch_size: int,
):
    opt = t.optim.Adam(actor.actor_parameters(), lr=lr)
    pbar = tqdm(range(n_epochs), desc="Fitting actor")
    first_loss = None
    for _ in pbar:
        np.random.shuffle(dataset)
        batch = []
        last_loss = None
        for i, (state, action) in enumerate(dataset):
            dist = actor.actor(state)
            loss = -dist.log_prob(action)
            batch.append(loss)
            if len(batch) == batch_size or i == len(dataset) - 1:
                loss_batch = t.stack(batch).mean()
                opt.zero_grad()
                loss_batch.backward()
                opt.step()
                last_loss = loss_batch.item()
                if first_loss is None:
                    first_loss = last_loss
                batch = []
                pbar.set_postfix({"loss": f"{first_loss:.4f}->{last_loss:.4f}"})


class MountainCarPolicy(Actor):
    @typed
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 3)

    @typed
    def actor(self, x: Float[TT, "2"]):
        x = self.fc(x)
        x = F.relu(x)
        x = self.fc2(x)
        return t.distributions.Categorical(logits=x)

    @typed
    def actor_parameters(self) -> Iterable[nn.Parameter]:
        return self.parameters()


def test_fit():
    actor = MountainCarPolicy()
    dataset = [
        (t.tensor([0.0, 0.0]), t.tensor([0])),
        (t.tensor([0.2, 0.0]), t.tensor([0])),
        (t.tensor([1.0, 0.0]), t.tensor([1])),
        (t.tensor([1.2, 0.0]), t.tensor([1])),
        (t.tensor([2.0, 0.0]), t.tensor([2])),
        (t.tensor([2.2, 0.0]), t.tensor([2])),
    ]
    fit_actor(actor, dataset, lr=1e-3, n_epochs=1000, batch_size=10)

    for x in t.linspace(0.0, 3.0, 10):
        state = t.tensor([x, 0.0])
        dist = actor.actor(state)
        print(x, dist.logits)


if __name__ == "__main__":
    test_fit()
