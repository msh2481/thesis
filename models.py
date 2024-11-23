from abc import ABC, abstractmethod

import torch as t
from beartype import beartype as typed
from envs.diff_stock_trading_env import DiffStockTradingEnv
from jaxtyping import Float, Int
from matplotlib import pyplot as plt
from torch import nn, Tensor as TT
from torch.distributions import Distribution
from torch.nn import functional as F
from tqdm.auto import tqdm


class SFTPolicy(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def action_distribution(self, state) -> Distribution:
        raise NotImplementedError()

    @typed
    def predict(
        self, state: Float[TT, "state_dim"], deterministic: bool = False
    ) -> Float[TT, "action_dim"]:
        dist = self.action_distribution(state)
        if deterministic:
            return dist.mode
        else:
            return dist.rsample()


class MLPPolicy(SFTPolicy):
    def __init__(self, input_dim: int, output_dim: int, dims: list[int] = [128, 128]):
        super().__init__()
        layers = []
        last_dim = input_dim
        for dim in dims:
            layers.append(nn.Linear(last_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(dim))
            last_dim = dim
        self.backbone = nn.Sequential(*layers)
        self.mu_head = nn.Linear(last_dim, output_dim)
        self.sigma_head = nn.Linear(last_dim, output_dim)

        nn.init.normal_(self.mu_head.weight, std=1e-9)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.normal_(self.sigma_head.weight, std=1e-9)
        self.sigma_head.bias.data.fill_(-100.0)

    @typed
    def action_distribution(self, state) -> Distribution:
        cur = state[1]
        nxt = state[3]
        diff = nxt - cur
        cutoff = 1.0
        x = self.backbone(state)
        mu = self.mu_head(x)
        sigma = F.softplus(self.sigma_head(x))
        eps = 1e-3
        if diff > cutoff:
            mu.data.fill_(eps)
            sigma.data.fill_(1e-9)
        elif diff < -cutoff:
            mu.data.fill_(-eps)
            sigma.data.fill_(1e-9)
        else:
            mu.data.fill_(0.0)
            sigma.data.fill_(1e-9)
        return t.distributions.Normal(mu, sigma)


def test_gradients():
    # mu gradients should be the same, sigma gradients should be None for mode
    policy = MLPPolicy(2, 2)
    opt = t.optim.Adam(policy.parameters(), lr=1e-3)
    state = t.randn(2)
    # test stochastic
    action = policy.predict(state, deterministic=False)
    print(action)
    loss = action.sum()
    loss.backward()
    print(policy.mu_head.weight.grad.flatten()[:10])
    print(policy.sigma_head.weight.grad.flatten()[:10])
    opt.zero_grad()
    # test deterministic
    action = policy.predict(state, deterministic=True)
    print(action)
    loss = action.sum()
    loss.backward()
    print(policy.mu_head.weight.grad.flatten()[:10])
    print(policy.sigma_head.weight.grad)


@typed
def rollout(
    policy: SFTPolicy,
    env: DiffStockTradingEnv,
    deterministic: bool = False,
) -> Float[TT, ""]:
    state, _ = env.reset_t()
    rewards = []
    for _ in range(env.max_step):
        action = policy.predict(state, deterministic=deterministic)
        state, reward, terminated, truncated, _ = env.step_t(action)
        rewards.append(reward)
        if terminated or truncated:
            break
    episode_return = sum(rewards)
    return episode_return


def fit_mlp_policy(
    env: DiffStockTradingEnv,
    n_epochs: int = 1000,
    batch_size: int = 1,
    lr: float = 1e-3,
):
    policy = MLPPolicy(env.state_dim, env.action_dim)
    opt = t.optim.Adam(policy.parameters(), lr=lr)
    pbar = tqdm(range(n_epochs))
    returns = []
    return_stds = []

    for it in pbar:
        episode_returns = []
        for _ in range(batch_size):
            episode_return = rollout(policy, env)
            episode_returns.append(episode_return)
        mean_return = sum(episode_returns) / batch_size
        return_tensors = t.tensor(episode_returns)
        std_return = return_tensors.detach().std()
        pbar.set_postfix(mean_return=mean_return.item(), std_return=std_return.item())

        opt.zero_grad()
        (-mean_return).backward()
        opt.step()

        returns.append(mean_return.item())
        return_stds.append(std_return.item())

        if it % 10 == 0:
            l = t.tensor(returns) - t.tensor(return_stds)
            r = t.tensor(returns) + t.tensor(return_stds)
            plt.clf()
            plt.axhline(0, color="k", linestyle="--")
            plt.fill_between(range(len(returns)), l, r, alpha=0.2)
            plt.plot(returns)
            plt.ylim(-1, 1)
            plt.savefig("logs/returns.png")

    return policy


if __name__ == "__main__":
    test_gradients()
