import numpy as np
import torch as t
from beartype import beartype as typed
from beartype.typing import Callable
from envs.diff_stock_trading_env import DiffStockTradingEnv
from jaxtyping import Float, Int
from loguru import logger
from matplotlib import pyplot as plt
from optimizers import CautiousAdamW, CautiousLion, CautiousSGD
from torch import nn, Tensor as TT
from tqdm.auto import tqdm
from layers import *


class PolyakNormalizer(nn.Module):
    def __init__(self, shape: tuple[int, ...]):
        super().__init__()
        self.register_buffer("mean", t.zeros(shape))
        # Second moment around mean
        self.register_buffer("M2", t.zeros(shape))
        self.register_buffer("count", t.tensor(0.0))

    @property
    def std(self) -> TT:
        return (
            t.sqrt(self.M2 / (self.count - 1) + 1e-8)
            if self.count > 1
            else t.ones_like(self.mean)
        )

    @typed
    def partial_fit(self, batch: Float[TT, "batch_size ..."]) -> None:
        # Drop samples with nan
        initial_size = batch.size(0)
        batch = batch[~t.isnan(batch).any(dim=tuple(range(1, batch.ndim)))]
        new_size = batch.size(0)
        if new_size < initial_size:
            logger.warning(
                f"Dropped {initial_size - new_size} samples with NaN out of {batch.shape} ({new_size} left)"
            )
        if not new_size:
            return
        assert not t.isnan(batch).any()

        with t.no_grad():
            batch_mean = batch.mean(dim=0)
            batch_size = t.tensor(batch.size(0), dtype=t.float32)

            # Compute batch M2
            batch_M2 = (batch - batch_mean).square().sum(dim=0)

            # Chan's parallel algorithm
            combined_size = self.count + batch_size
            delta = batch_mean - self.mean
            self.mean = self.mean + delta * (batch_size / combined_size)
            self.M2 = (
                self.M2
                + batch_M2
                + delta * delta * (self.count * batch_size / combined_size)
            )
            self.count = combined_size

    @typed
    def transform(
        self, batch: Float[TT, "batch_size ..."]
    ) -> Float[TT, "batch_size ..."]:
        mu = self.mean.unsqueeze(0)
        sigma = self.std.unsqueeze(0)
        assert not t.isnan(mu).any(), f"NaN mean: {mu.shape}"
        assert not t.isnan(sigma).any(), f"NaN std: {sigma.shape}"
        return (batch - mu) / sigma

    @typed
    def fit_transform(
        self, batch: Float[TT, "batch_size ..."]
    ) -> Float[TT, "batch_size ..."]:
        self.partial_fit(batch)
        return self.transform(batch)


class EMA:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.values = {}

    def update(self, key: str, value: Float[TT, ""]) -> None:
        self.values[key] = self.alpha * self[key] + (1 - self.alpha) * value

    def __getitem__(self, key: str) -> Float[TT, ""]:
        return self.values.get(key, t.zeros(()))


def product(shape: tuple[int, ...]) -> int:
    result = 1
    for dim in shape:
        result *= dim
    return result


class MLPPolicy(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, ...],
        output_dim: int,
        dropout_rate: float = 0.0,
        n_layers: int = 1,
    ):
        super().__init__()
        self.normalizer = PolyakNormalizer(input_shape)
        self.register_buffer("dropout_rate", t.tensor(dropout_rate))
        layers = []
        stock_dim, tech_dim = input_shape
        for _ in range(n_layers):
            layers.append(StockBlock(stock_dim, tech_dim, tech_dim, True))
            layers.append(nn.Dropout(self.dropout_rate))
        layers.append(DimWiseLinear(tech_dim, 1, dim=1, bias=True, gain=1e-3))
        layers.append(nn.Flatten())
        self.backbone = nn.Sequential(*layers)

    @typed
    def predict(
        self, state: Float[TT, "..."], noise_level: float
    ) -> Float[TT, "output_dim"]:
        return self.predict_batch(state.unsqueeze(0), noise_level).squeeze(0)

    @typed
    def predict_batch(
        self, state: Float[TT, "batch ..."], noise_level: float
    ) -> Float[TT, "batch output_dim"]:
        with t.no_grad():
            for p in self.parameters():
                if t.isnan(p.data).any():
                    logger.warning(f"NaN detected in parameter: {p}")
                    p.data = t.where(t.isnan(p.data), t.zeros_like(p.data), p.data)
                p.data.clamp_(-10, 10)

        if t.isnan(state).any():
            logger.warning(f"NaN state: {state.shape}")
        if self.training:
            state_normalized = self.normalizer.fit_transform(state)
        else:
            state_normalized = self.normalizer.transform(state)
        if t.isnan(state_normalized).any():
            logger.warning(f"NaN state_normalized: {state_normalized.shape}")
        logits = self.backbone(state_normalized)
        logits.data.clamp_(-5, 5)
        unnormalized_position = (logits + noise_level * t.randn_like(logits)).exp()
        position = unnormalized_position / (
            unnormalized_position.sum(dim=1, keepdim=True) + 1
        )
        return position


def test_gradients():
    # mu gradients should be the same, sigma gradients should be None for mode
    policy = MLPPolicy((2, 2), 2)
    opt = t.optim.Adam(policy.parameters(), lr=1e-3)
    n_steps = 100

    grad_history = {}
    for name, _ in policy.named_parameters():
        grad_history[name] = []
    true_action = t.tensor([0.0, 0.33])

    for step in range(n_steps):
        # Generate random state and target each time
        state = t.randn(2, 2)

        # Test stochastic
        action = policy.predict(state, noise_level=1.0)
        loss = (action - true_action).square().sum()
        loss.backward()

        # Track gradients
        for name, param in policy.named_parameters():
            if param.grad is not None:
                assert not t.isnan(
                    param.grad
                ).any(), f"NaN detected in {name} gradients at step {step}"
                grad_history[name].append(param.grad.norm().item())

        opt.step()
        opt.zero_grad()

        # Test deterministic
        action = policy.predict(state, noise_level=0.0)
        loss = (action - true_action).square().sum()
        loss.backward()

        for name, param in policy.named_parameters():
            if param.grad is not None:
                assert not t.isnan(
                    param.grad
                ).any(), f"NaN detected in {name} gradients at step {step}"

        opt.step()
        opt.zero_grad()

        if step % 10 == 0:
            logger.info(f"Step {step}, Loss: {loss.item():.3f}")
            for name, grads in grad_history.items():
                if grads:
                    logger.info(f"{name} grad norm: {grads[-1]:.3e}")

    # Plot gradient norms over time
    plt.figure(figsize=(10, 5))
    for name, grads in grad_history.items():
        if grads:
            plt.plot(grads, label=name)
    plt.yscale("log")
    plt.xlabel("Step")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Norms During Training")
    plt.legend()
    plt.grid(True)
    plt.savefig("logs/gradient_test.png")
    plt.close()


@typed
def rollout(
    policy: MLPPolicy,
    env: DiffStockTradingEnv,
    noise_level: float,
) -> Float[TT, ""]:
    state, _ = env.reset()
    zero_with_gradients = policy.predict(state, noise_level).sum() * 0
    rewards = [zero_with_gradients]
    total_reward = 0
    for step in range(env.n_steps):
        action = policy.predict(state, noise_level)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward.item()
        rewards.append(reward)
        if terminated or truncated:
            break
    episode_return = sum(rewards)
    return episode_return


@typed
def plot_returns(
    returns: list[float],
    returns_ema: list[float],
    return_stds: list[float],
    val_returns: list[float] | None = None,
    val_returns_ema: list[float] | None = None,
    val_period: int = 5,
) -> None:
    """Plot training and validation returns."""
    l = t.tensor(returns) - t.tensor(return_stds)
    r = t.tensor(returns) + t.tensor(return_stds)
    plt.close()
    plt.clf()

    n_plots = 2 if val_returns else 1

    plt.figure(figsize=(10, 10))
    # Training returns plot
    plt.subplot(n_plots, 1, 1)
    plt.title("Training Returns")
    plt.axhline(0, color="k", linestyle="--")
    plt.grid(True, which="major", alpha=0.5)
    plt.grid(True, which="minor", alpha=0.1)
    plt.minorticks_on()
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    plt.fill_between(range(len(returns)), l, r, alpha=0.2, color="red")
    plt.plot(returns, "r-", lw=0.5, label="Current Policy")
    plt.plot(returns_ema, "r--", label="Current EMA")
    plt.legend()
    returns_95 = t.quantile(t.tensor(returns), 0.95)
    returns_5 = t.quantile(t.tensor(returns), 0.05)
    plt.ylim(returns_5 - 0.1, returns_95 + 0.1)

    # Validation returns plot if applicable
    if val_returns:
        plt.subplot(n_plots, 1, 2)
        plt.title("Validation Returns")
        plt.grid(True, which="major", alpha=0.5)
        plt.grid(True, which="minor", alpha=0.1)
        plt.minorticks_on()
        plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.1))
        val_x = list(range(0, len(returns), val_period))
        plt.plot(val_x, val_returns, "r-", lw=0.5, label="Current Policy")
        plt.plot(val_x, val_returns_ema, "r--", label="Current EMA")
        plt.legend()
        val_returns_95 = t.quantile(t.tensor(val_returns), 0.95)
        val_returns_5 = t.quantile(t.tensor(val_returns), 0.05)
        plt.ylim(val_returns_5 - 0.1, val_returns_95 + 0.1)
        plt.axhline(0, color="k", linestyle="--")

    plt.tight_layout()
    plt.savefig("logs/info.png")
    plt.close()


@typed
def polyak_update(
    avg_state: dict, prev_state: dict, current_state: dict, it: int
) -> tuple[dict, dict]:
    for key in avg_state:
        if key.startswith("normalizer"):
            # Copy current model's normalizer stats directly
            avg_state[key] = current_state[key].clone()
            continue

        n = t.tensor(it + 1, dtype=t.float32)
        alpha_avg = 0.5  # Hyperparameter for adaptive weighted averaging

        # Implement the adaptive weighted averaging formula
        term1 = (n / (n + 1)) * avg_state[key]
        term2 = ((1 - (n + 1) ** alpha_avg) / (n + 1)) * prev_state[key]
        term3 = (n + 1) ** (alpha_avg - 1) * current_state[key]

        avg_state[key] = term1 + term2 + term3

    # Store current state for next iteration
    prev_state = {k: v.clone() for k, v in current_state.items()}
    return avg_state, prev_state


@typed
def train_batch_pnl(
    env_factory: Callable[[], DiffStockTradingEnv],
    policy: MLPPolicy,
    rollout_fn: Callable[[MLPPolicy, DiffStockTradingEnv, float], Float[TT, ""]],
    batch_size: int,
    noise_level: float,
) -> tuple[Float[TT, ""], Float[TT, ""]]:
    env = env_factory()
    episode_returns = []
    for _ in range(batch_size):
        episode_return = rollout_fn(policy, env, noise_level)
        episode_returns.append(episode_return)

    return_tensors = t.tensor(episode_returns)
    mean_return = (sum(episode_returns) / batch_size).clamp(-100, None)
    std_return = (
        return_tensors.detach().std() if len(return_tensors) > 1 else t.tensor(0.0)
    )
    return mean_return, std_return


@typed
def fit_policy_on_pnl(
    env_factory: Callable[[], DiffStockTradingEnv],
    val_env_factory: Callable[[], DiffStockTradingEnv] | None = None,
    n_epochs: int = 1000,
    batch_size: int = 1,
    lr: float = 1e-3,
    val_period: int = 5,
    n_layers: int = 1,
    dropout_rate: float = 0.0,
    init_from: str | None = None,
) -> MLPPolicy:
    # Initialize environment and policy
    env = env_factory()
    policy = MLPPolicy(
        env.observation_space.shape,
        env.action_space.shape[0],
        dropout_rate=dropout_rate,
        n_layers=n_layers,
    )
    n_params = sum(p.numel() for p in policy.parameters())
    logger.warning(f"#params = {n_params}")
    if init_from is not None:
        policy.load_state_dict(t.load(init_from))

    opt = CautiousAdamW(policy.parameters(), lr=lr)
    scheduler = t.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.95)

    pbar = tqdm(range(n_epochs))
    train_returns = []
    train_stds = []
    train_returns_ema = []
    val_returns = []
    val_returns_ema = []

    ema = EMA(alpha=0.9)

    for epoch in pbar:
        # Training
        policy.train()
        mean_return, std_return = train_batch_pnl(
            env_factory, policy, rollout, batch_size, noise_level=0.0
        )
        train_returns.append(mean_return.item())
        train_stds.append(std_return.item())
        ema.update("train_return", mean_return)
        train_returns_ema.append(ema["train_return"].item())

        # Compute gradients and update
        (-mean_return).backward()
        opt.step()
        opt.zero_grad()
        scheduler.step()

        # Validation
        if val_env_factory is not None and epoch % val_period == 0:
            policy.eval()
            with t.no_grad():
                val_return, _ = train_batch_pnl(
                    val_env_factory, policy, rollout, batch_size, noise_level=0.0
                )
                val_returns.append(val_return.item())
                ema.update("val_return", val_return)
                val_returns_ema.append(ema["val_return"].item())

        # Update progress bar
        postfix = {
            "train_return": f"{mean_return.item():.3e}",
            "train_std": f"{std_return.item():.3e}",
        }
        if val_returns:
            postfix["val_return"] = f"{val_returns[-1]:.3e}"
        pbar.set_postfix(**postfix)

        # Save model and plot periodically
        if epoch % 10 == 0:
            t.save(policy.state_dict(), f"checkpoints/policy_pnl_{epoch}.pth")

            plot_returns(
                returns=train_returns,
                returns_ema=train_returns_ema,
                return_stds=train_stds,
                val_returns=val_returns,
                val_returns_ema=val_returns_ema,
                val_period=val_period,
            )

    return policy


@typed
def fit_policy_on_optimal(
    policy: MLPPolicy,
    n_epochs: int,
    batch_size: int,
    lr: float,
    full_train_env: DiffStockTradingEnv,
    full_val_env: DiffStockTradingEnv | None = None,
    val_length: int = 1000,
):
    n_params = sum(p.numel() for p in policy.parameters())
    logger.debug(f"policy:\n{policy}")
    logger.warning(f"#params = {n_params}")

    X_train, y_train = env_to_dataset(full_train_env)
    train_dataset = t.utils.data.TensorDataset(X_train, y_train)
    train_loader = t.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    X_val, y_val = None, None
    val_loader = None
    if full_val_env is not None:
        X_val, y_val = env_to_dataset(full_val_env)
        val_dataset = t.utils.data.TensorDataset(X_val, y_val)
        val_loader = t.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

    opt = CautiousAdamW(policy.parameters(), lr=lr)
    scheduler = t.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.95)

    pbar = tqdm(range(n_epochs))
    train_losses = []
    val_losses = []
    train_pnls = []
    val_pnls = []

    def eval_pnl(env):
        policy.eval()
        with t.no_grad():
            return rollout(policy, env, noise_level=0.0).item()

    for epoch in pbar:
        # Training
        policy.train()
        epoch_losses = []

        for X_batch, y_batch in train_loader:
            opt.zero_grad()
            pred = policy.predict_batch(X_batch, noise_level=0.0)
            loss = t.nn.functional.mse_loss(pred, y_batch)
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())

        train_loss = sum(epoch_losses) / len(epoch_losses)
        train_losses.append(train_loss)

        # Validation
        if val_loader is not None:
            policy.eval()
            val_epoch_losses = []
            with t.no_grad():
                for X_batch, y_batch in val_loader:
                    pred = policy.predict_batch(X_batch, noise_level=0.0)
                    loss = t.nn.functional.mse_loss(pred, y_batch)
                    val_epoch_losses.append(loss.item())
            val_loss = sum(val_epoch_losses) / len(val_epoch_losses)
            val_losses.append(val_loss)

        scheduler.step()

        # Evaluate PnL metrics
        if epoch % 50 == 0:
            train_pnl = eval_pnl(full_train_env.subsegment(0, val_length))
            train_pnls.append(train_pnl)
            if full_val_env is not None:
                val_pnl = eval_pnl(full_val_env.subsegment(0, val_length))
                val_pnls.append(val_pnl)

        # Update progress bar
        postfix = {
            "train_loss": f"{train_loss:.3e}",
            "train_pnl": f"{train_pnl:.3e}",
        }
        if val_losses:
            postfix["val_loss"] = f"{val_losses[-1]:.3e}"
        if val_pnl is not None:
            postfix["val_pnl"] = f"{val_pnl:.3e}"
        pbar.set_postfix(**postfix)

        # Save model periodically
        if epoch % 10 == 0:
            t.save(policy.state_dict(), f"checkpoints/policy_supervised_{epoch}.pth")

            # Plot losses
            plt.figure(figsize=(10, 5))
            plt.subplot(2, 1, 1)
            plt.plot(np.arange(len(train_losses)), train_losses, label="Train Loss")
            if val_losses:
                plt.plot(np.arange(len(val_losses)), val_losses, label="Val Loss")
            plt.yscale("log")
            plt.grid(True)
            plt.legend()
            plt.title("Training and Validation Losses")
            plt.tight_layout()
            plt.subplot(2, 1, 2)
            plt.plot(np.arange(len(train_pnls)) * 50, train_pnls, label="Train PnL")
            if val_pnls:
                plt.plot(np.arange(len(val_pnls)) * 50, val_pnls, label="Val PnL")
            plt.grid(True)
            plt.legend()
            plt.title("Training and Validation PnL")
            plt.tight_layout()
            plt.savefig("logs/supervised_training.png")
            plt.close()

    return policy


def test_normalizer():
    # Setup
    n_features = 3
    normalizer = PolyakNormalizer(n_features)

    # Generate data from different normal distributions for each feature
    means = t.tensor([1.0, -2.0, 5.0])
    stds = t.tensor([0.5, 2.0, 3.0])
    n_batches = 1000
    batch_size = 5

    # Training loop
    pbar = tqdm(range(n_batches), desc="Training normalizer")
    for _ in pbar:
        # Generate batch from factorized normal
        batch = t.randn(batch_size, n_features) * stds + means
        normalized = normalizer.fit_transform(batch)

        # Compute statistics of normalized data
        with t.no_grad():
            actual_mean = normalized.mean(dim=0)
            actual_std = normalized.std(dim=0)
            mean_error = (actual_mean - 0.0).abs().mean()
            std_error = (actual_std - 1.0).abs().mean()
            pbar.set_postfix(
                mean_error=f"{mean_error:.3f}",
                std_error=f"{std_error:.3f}",
            )

    # Final test on fresh data
    with t.no_grad():
        test_batch = t.randn(1000, n_features) * stds + means
        normalized = normalizer.transform(test_batch)
        final_mean = normalized.mean(dim=0)
        final_std = normalized.std(dim=0)

        print("\nFinal statistics:")
        print(f"Mean (should be ~0): {final_mean}")
        print(f"Std (should be ~1): {final_std}")
        print(f"Learned means: {normalizer.mean}")
        print(f"Learned stds: {normalizer.std}")

        # Assert the normalization worked
        assert t.all(final_mean.abs() < 0.1), "Means not close enough to 0"
        assert t.all((final_std - 1.0).abs() < 0.1), "Stds not close enough to 1"


@typed
def env_to_dataset(
    env: DiffStockTradingEnv,
) -> tuple[Float[TT, "..."], Float[TT, "..."]]:
    states = []
    y = env.get_optimal_positions()
    state, _ = env.reset()
    states.append(state.clone())
    for action in y:
        state, reward, terminated, truncated, _ = env.step(action)
        states.append(state.clone())
        if terminated or truncated:
            break
    states = t.stack(states)
    logger.warning(f"States shape: {states.shape} | Targets shape: {y.shape}")
    logger.warning(f"State std by column: {states.std(dim=0)}")
    logger.warning(f"Target std by column: {y.std(dim=0)}")
    # sample_indices = t.randint(0, states.size(0), (10,))
    # for ind in sample_indices:
    #     logger.debug("--------------------------------")
    #     logger.debug(f"Target {ind}: {y[ind].numpy()}")
    #     logger.debug(f"State {ind}: {states[ind].numpy()}")
    X_nan = t.isnan(states).any(dim=1)
    Y_nan = t.isnan(y)
    if X_nan.any():
        logger.warning(f"NaN states: {X_nan.sum()}. Indices: {X_nan.nonzero()}")
    if Y_nan.any():
        logger.warning(f"NaN y: {Y_nan.sum()}. Indices: {Y_nan.nonzero()}")

    return states[:-1], y


if __name__ == "__main__":
    test_gradients()
