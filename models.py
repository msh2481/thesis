from abc import ABC, abstractmethod

import torch as t
from beartype import beartype as typed
from beartype.typing import Callable
from envs.benchmark import PredictableEnv
from envs.diff_stock_trading_env import DiffStockTradingEnv
from jaxtyping import Float, Int
from loguru import logger
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


class TruePolicy(SFTPolicy):
    def __init__(self, input_dim: int, output_dim: int, dims: list[int] = [128, 128]):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    @typed
    def action_distribution(self, state) -> Distribution:
        cash_ratio = state[0]
        cur = state[1]
        stock = state[2]
        nxt = state[3]
        cash = state[4]
        cash_div_cur = state[5]
        mu = t.zeros(self.output_dim)
        sigma = t.zeros(self.output_dim)
        if nxt > cur and cash_ratio > 0.3:
            mu.data.fill_(0.2 * cash_div_cur)
        elif nxt < cur and cash_ratio < 0.7:
            mu.data.fill_(-0.2 * stock)
        sigma.data.fill_(1e-9)
        return t.distributions.Normal(mu, sigma)


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
        for layer in layers:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                nn.init.zeros_(layer.bias)
        self.mu_head = nn.Linear(last_dim, output_dim)
        self.sigma_head = nn.Linear(last_dim, output_dim)

        nn.init.normal_(self.mu_head.weight, std=1e-9)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.normal_(self.sigma_head.weight, std=1e-9)
        self.sigma_head.bias.data.fill_(-10.0)

    @typed
    def action_distribution(self, state) -> Distribution:
        # cur = state[1]
        # nxt = state[3]
        # diff = cur - nxt
        # cutoff = 0.1
        x = self.backbone(state)
        mu = self.mu_head(x)
        sigma = F.softplus(self.sigma_head(x)) + 1e-9
        # eps = 1.0
        # if diff > cutoff:
        #     mu.data.fill_(eps)
        #     sigma.data.fill_(1e-9)
        # elif diff < -cutoff:
        #     mu.data.fill_(-eps)
        #     sigma.data.fill_(1e-9)
        # else:
        #     mu.data.fill_(0.0)
        #     sigma.data.fill_(1e-9)
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
    zero_with_gradients = policy.predict(state).sum() * 0
    rewards = [zero_with_gradients]
    total_reward = 0
    total_penalty = 0
    for step in range(env.max_step):
        action = policy.predict(state, deterministic=deterministic)
        state, reward, terminated, truncated, _ = env.step_t(action)
        penalty = env._get_penalty()
        total_reward += reward.item()
        total_penalty += penalty.item()
        rewards.append(reward)
        if terminated or truncated:
            break
        if total_penalty > max(total_reward, 0.1):
            # logger.warning(f"Penalty: {total_penalty}")
            break
    episode_return = sum(rewards)
    return episode_return


@typed
def imitation_rollout(
    policy: SFTPolicy,
    env: DiffStockTradingEnv,
    deterministic: bool = False,
    use_env: bool = True,
) -> Float[TT, ""]:
    true_policy = TruePolicy(env.state_dim, env.action_dim)
    rewards = []
    if use_env:
        state, _ = env.reset_t()
        for _ in range(env.max_step):
            action = policy.predict(state, deterministic=deterministic)
            true_action = true_policy.predict(state, deterministic=deterministic)
            state, reward, terminated, truncated, _ = env.step_t(true_action)
            loss = (action - true_action).square().sum()
            # logger.debug(
            #     f"action: {action.item():.2f}, true_action: {true_action.item():.2f}, loss: {loss.item():.2f}"
            # )
            rewards.append(-loss)
            if terminated or truncated:
                break
        episode_return = sum(rewards) / len(rewards)
        return episode_return
    else:
        for _ in range(50):
            state = t.randn(env.state_dim).exp()
            action = policy.predict(state, deterministic=deterministic)
            true_action = true_policy.predict(state, deterministic=deterministic)
            loss = (action - true_action).square().sum()
            rewards.append(-loss)
        episode_return = sum(rewards) / len(rewards)
        return episode_return


def fit_mlp_policy(
    env: DiffStockTradingEnv,
    n_epochs: int = 1000,
    batch_size: int = 1,
    lr: float = 1e-3,
    rollout_fn: Callable[[SFTPolicy, DiffStockTradingEnv], Float[TT, ""]] = rollout,
    init_from: str | None = None,
):
    policy = MLPPolicy(env.state_dim, env.action_dim)
    if init_from is not None:
        policy.load_state_dict(t.load(init_from, weights_only=False))
    opt = t.optim.Adam(policy.parameters(), lr=lr)
    scheduler = t.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.95)
    pbar = tqdm(range(n_epochs))
    returns = []
    return_stds = []
    returns_ema = []
    current_ema = 0.0
    alpha = 0.95
    gradients = []

    for it in pbar:
        episode_returns = []
        for _ in range(batch_size):
            episode_return = rollout_fn(policy, env)
            episode_returns.append(episode_return)
        mean_return = sum(episode_returns) / batch_size
        return_tensors = t.tensor(episode_returns)
        std_return = return_tensors.detach().std()
        pbar.set_postfix(
            mean=mean_return.item(),
            std=std_return.item(),
            ema=current_ema,
        )

        opt.zero_grad()
        (-mean_return).backward()
        # Compute l2 norm across all parameters
        max_gradient = t.sqrt(
            sum(p.grad.flatten().norm().square() for p in policy.parameters())
        )
        # Normalize each parameter's gradients by dividing by total norm
        if max_gradient > 0:
            for p in policy.parameters():
                if p.grad is not None:
                    p.grad.div_(max_gradient)

        opt.step()
        scheduler.step()

        returns.append(mean_return.item())
        current_ema = alpha * current_ema + (1 - alpha) * mean_return.item()
        returns_ema.append(current_ema)
        return_stds.append(std_return.item())
        gradients.append(max_gradient)

        if it % 10 == 0:
            l = t.tensor(returns) - t.tensor(return_stds)
            r = t.tensor(returns) + t.tensor(return_stds)
            plt.clf()
            plt.subplot(2, 1, 1)
            plt.axhline(0, color="k", linestyle="--")
            plt.fill_between(range(len(returns)), l, r, alpha=0.2)
            plt.plot(returns, "r", label="returns")
            plt.plot(returns_ema, "b--", label="EMA")
            plt.legend()
            returns_95 = t.quantile(t.tensor(returns), 0.95)
            returns_5 = t.quantile(t.tensor(returns), 0.05)
            plt.ylim(returns_5 - 0.1, returns_95 + 0.1)

            plt.subplot(2, 1, 2)
            plt.plot(gradients)
            plt.yscale("log")
            plt.savefig("logs/info.png")

            t.save(policy.state_dict(), f"checkpoints/policy_{it}.pth")

    return policy


def test_gradients():

    def helper(i: int, n: int) -> Float[TT, "n"]:
        env = PredictableEnv.create(1, 1, n + 2)
        state, _ = env.reset_t()
        policy = MLPPolicy(env.state_dim, env.action_dim)
        xs = []
        y = None
        for step in range(n):
            action = policy.predict(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step_t(action)
            assert not (terminated or truncated)
            xs.append(state)
            if i == step:
                y = reward
        for x in xs:
            x.retain_grad()

        y.backward()
        gradients = t.tensor(
            [x.grad.sum().item() if x.grad is not None else 0.0 for x in xs]
        )
        return gradients

    n = 10
    gradient_map = t.zeros(n, n)
    for i in range(n):
        gradient_map[i] = helper(i, n)

    plt.figure(figsize=(8, 6))
    plt.imshow(gradient_map.abs(), norm="log", cmap="viridis")
    plt.colorbar(label="Gradient Magnitude")

    # Add text annotations for gradient values in each cell
    for i in range(gradient_map.shape[0]):
        for j in range(gradient_map.shape[1]):
            plt.text(
                j,
                i,
                f"{gradient_map[i,j].item():.2e}",
                ha="center",
                va="center",
                color="white",
                size=6,
            )

    plt.xlabel("Action Step")
    plt.ylabel("Reward Step")
    plt.title("Gradient Flow Map")
    plt.savefig("logs/gradient_map.png")
    plt.close()


def test_penalty():
    env = PredictableEnv.create(1, 1, 100)
    state, _ = env.reset_t()
    action = t.tensor([-1.5])
    state, reward, terminated, truncated, _ = env.step_t(action)
    print(env._get_penalty().item())


if __name__ == "__main__":
    test_penalty()
