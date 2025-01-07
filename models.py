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
            logger.warning(f"Dropped {initial_size - new_size} samples with NaN")
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
        return (batch - self.mean.unsqueeze(0)) / (self.std.unsqueeze(0))

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
    ):
        super().__init__()
        self.normalizer = PolyakNormalizer(input_shape)
        self.register_buffer("dropout_rate", t.tensor(dropout_rate))
        layers = []
        n_mid, m_mid = 8, 8
        pre_head = 64
        layers.append(DimWisePair(*input_shape, n_mid, m_mid))
        for _ in range(1):
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
            layers.append(DimWisePair(n_mid, m_mid, n_mid, m_mid))
        last_dim = n_mid * m_mid
        layers.append(nn.Flatten())
        layers.append(nn.Linear(last_dim, pre_head))
        layers.append(nn.ReLU())
        self.backbone = nn.Sequential(*layers)
        for layer in layers:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                nn.init.zeros_(layer.bias)
        self.head = nn.Linear(pre_head, output_dim)
        nn.init.normal_(self.head.weight, std=1e-9)
        nn.init.zeros_(self.head.bias)

    @typed
    def predict(
        self, state: Float[TT, "..."], noise_level: float
    ) -> Float[TT, "output_dim"]:
        with t.no_grad():
            for p in self.parameters():
                if t.isnan(p.data).any():
                    logger.warning(f"NaN detected in parameter: {p}")
                    p.data = t.where(t.isnan(p.data), t.zeros_like(p.data), p.data)
                p.data.clamp_(-10, 10)

        if self.training:
            state_normalized = self.normalizer.fit_transform(state.unsqueeze(0))
        else:
            state_normalized = self.normalizer.transform(state.unsqueeze(0))
        x = self.backbone(state_normalized)
        logits = self.head(x).squeeze(0)
        logits.data.clamp_(-5, 5)
        unnormalized_position = (logits + noise_level * t.randn_like(logits)).exp()
        position = unnormalized_position / (unnormalized_position.sum() + 1)
        return position


def test_gradients():
    # mu gradients should be the same, sigma gradients should be None for mode
    policy = MLPPolicy(2, 2)
    opt = t.optim.Adam(policy.parameters(), lr=1e-3)
    state = t.randn(2)
    # test stochastic
    action = policy.predict(state, noise_level=1.0)
    print(action)
    loss = action.sum()
    loss.backward()
    print(policy.mu_head.weight.grad.flatten()[:10])
    print(policy.sigma_head.weight.grad.flatten()[:10])
    opt.zero_grad()
    # test deterministic
    action = policy.predict(state, noise_level=0.0)
    print(action)
    loss = action.sum()
    loss.backward()
    print(policy.mu_head.weight.grad.flatten()[:10])
    print(policy.sigma_head.weight.grad)


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
    gradients: list[float],
    val_returns: list[float] | None = None,
    val_returns_ema: list[float] | None = None,
    val_period: int = 5,
    eval_interval: int = 1,
    avg_returns: list[float] | None = None,
    avg_returns_ema: list[float] | None = None,
    val_avg_returns: list[float] | None = None,
    val_avg_returns_ema: list[float] | None = None,
) -> None:
    """Plot training and validation returns."""
    l = t.tensor(returns) - t.tensor(return_stds)
    r = t.tensor(returns) + t.tensor(return_stds)
    plt.close()
    plt.clf()

    n_plots = 3 if val_returns else 2

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
    if avg_returns is not None:
        eval_x = list(range(0, len(returns), eval_interval))
        plt.plot(eval_x, avg_returns, "b-", lw=0.5, label="Polyak Average")
        plt.plot(eval_x, avg_returns_ema, "b--", label="Polyak EMA")
    plt.legend()
    returns_95 = t.quantile(t.tensor(returns), 0.95)
    returns_5 = t.quantile(t.tensor(returns), 0.05)
    plt.ylim(returns_5 - 0.1, returns_95 + 0.1)

    # Gradient plot
    plt.subplot(n_plots, 1, 2)
    plt.title("Gradient Norms")
    plt.plot(gradients)
    plt.yscale("log")
    plt.grid(True, which="major", alpha=0.3)

    # Validation returns plot if applicable
    if val_returns:
        plt.subplot(n_plots, 1, 3)
        plt.title("Validation Returns")
        plt.grid(True, which="major", alpha=0.5)
        plt.grid(True, which="minor", alpha=0.1)
        plt.minorticks_on()
        plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.1))
        val_x = list(range(0, len(returns), val_period))
        plt.plot(val_x, val_returns, "r-", lw=0.5, label="Current Policy")
        plt.plot(val_x, val_returns_ema, "r--", label="Current EMA")
        if val_avg_returns is not None:
            plt.plot(val_x, val_avg_returns, "b-", lw=0.5, label="Polyak Average")
            plt.plot(val_x, val_avg_returns_ema, "b--", label="Polyak EMA")
        plt.legend()
        if val_returns:
            val_returns_95 = t.quantile(t.tensor(val_returns), 0.95)
            val_returns_5 = t.quantile(t.tensor(val_returns), 0.05)
            plt.ylim(val_returns_5 - 0.1, val_returns_95 + 0.1)
        plt.axhline(0, color="k", linestyle="--")

    plt.tight_layout()
    plt.savefig("logs/info.png")
    plt.close()


@typed
def fit_policy(
    env_factory: Callable[[], DiffStockTradingEnv],
    n_epochs: int = 1000,
    batch_size: int = 1,
    lr: float = 1e-3,
    rollout_fn: Callable[
        [MLPPolicy, DiffStockTradingEnv, float], Float[TT, ""]
    ] = rollout,
    init_from: str | None = None,
    polyak_average: bool = False,
    eval_interval: int = 5,
    max_weight: float = 10.0,
    langevin_coef: float = 0.0,
    val_env_factory: Callable[[], DiffStockTradingEnv] | None = None,
    val_period: int = 10,
    prior_std: float = 1e9,
    dropout_rate: float = 0.0,
    noise_level: float = 1e-4,
):
    # Get dimensions once from a temporary environment
    tmp_env = env_factory()
    state_shape, action_dim = (
        tmp_env.observation_space.shape,
        tmp_env.action_space.shape[0],
    )
    del tmp_env

    policy_kwargs = {
        "input_shape": state_shape,
        "output_dim": action_dim,
        "dropout_rate": dropout_rate,
    }
    policy = MLPPolicy(**policy_kwargs)
    n_params = sum(p.numel() for p in policy.parameters())
    logger.warning(f"MLPPolicy initialized with #parameters = {n_params}")
    if init_from is not None:
        policy.load_state_dict(t.load(init_from, weights_only=False))

    ema = EMA(alpha=0.95)

    # Initialize average state dict if needed
    avg_state = None
    prev_state = None
    if polyak_average:
        avg_state = policy.state_dict()
        prev_state = policy.state_dict()
        avg_returns = []
        avg_returns_ema = []
        last_avg_return = float("-inf")

    opt = CautiousAdamW(
        policy.parameters(),
        lr=lr,
    )

    scheduler = t.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.95)
    pbar = tqdm(range(n_epochs))
    returns = []
    return_stds = []
    returns_ema = []
    gradients = []

    # Initialize validation tracking if needed
    val_returns = []
    val_returns_ema = []
    val_avg_returns = []
    val_avg_returns_ema = []

    current_lr = lr

    for it in pbar:
        env = env_factory()
        episode_returns = []
        for _ in range(batch_size):
            episode_return = rollout_fn(policy, env, noise_level)
            episode_returns.append(episode_return)

        mean_return = (sum(episode_returns) / batch_size).clamp(-100, None)
        return_tensors = t.tensor(episode_returns)
        std_return = (
            return_tensors.detach().std() if len(return_tensors) > 1 else t.tensor(0.0)
        )

        pbar_info = {
            "cur": f"{mean_return.item():.3f}",
            "ema": f"{ema['cur']:3f}",
            "lr": f"{current_lr:.2e}",
            "val_ema": f"{ema['val']:3f}",
            "val_avg_ema": f"{ema['val_avg']:3f}",
        }
        if polyak_average:
            pbar_info.update(
                {
                    "avg": f"{last_avg_return:.3f}",
                    "avg_ema": f"{ema['avg']:3f}",
                }
            )
        pbar.set_postfix(**pbar_info)

        opt.zero_grad()
        regularization = 0.0
        for p in policy.parameters():
            regularization += (p / prior_std).square().sum()
        (regularization - mean_return).backward()

        # Gradient normalization
        sum_of_squares = t.tensor(1.0)
        count = 1.0
        for p in policy.parameters():
            g = p.grad
            if g is None:
                continue
            delta = g.flatten().square().sum()
            if not delta.isfinite():
                logger.warning(f"Gradient is NaN: {g}")
                p.grad = t.zeros_like(p.grad)
                g = p.grad
                delta = g.flatten().square().sum()
            assert delta.isfinite(), "Delta is NaN"
            sum_of_squares += delta
            count += g.numel()

        sum_of_squares = t.minimum(sum_of_squares, t.tensor(1e20))
        normalization_constant = t.sqrt(sum_of_squares / count)

        assert sum_of_squares.isfinite(), "Sum of squares is NaN"
        assert (sum_of_squares >= 0).all(), "Sum of squares is negative"
        assert t.tensor(count).isfinite(), "Count is NaN"
        assert count > 0, "Count is non-positive"
        assert (
            normalization_constant.isfinite()
        ), "Gradient normalization constant is NaN"
        # if normalization_constant > 0:
        #     for p in policy.parameters():
        #         if p.grad is not None:
        #             p.grad.div_(normalization_constant)

        opt.step()
        scheduler.step()
        current_lrs = scheduler.get_last_lr()
        assert (
            max(current_lrs) - min(current_lrs) < 1e-6
        ), "Learning rate is not constant"
        current_lr = current_lrs[0]

        C = langevin_coef * ((2 * current_lr) ** 0.5)
        for p in policy.parameters():
            p.data.add_(C * t.randn_like(p.data))

        for p in policy.parameters():
            p.data.clamp_(-max_weight, max_weight)

        # Update average state dict
        if polyak_average:
            with t.no_grad():
                current_state = policy.state_dict()
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

        # Evaluate averaged model periodically
        if polyak_average and it % eval_interval == 0:
            avg_policy = MLPPolicy(**policy_kwargs)
            avg_policy.load_state_dict(avg_state)
            avg_policy.eval()
            avg_episode_returns = []
            for _ in range(batch_size):
                avg_return = rollout_fn(avg_policy, env, noise_level)
                avg_episode_returns.append(avg_return)
            avg_mean_return = (sum(avg_episode_returns) / batch_size).clamp(-100, None)
            last_avg_return = avg_mean_return.item()
            avg_returns.append(last_avg_return)
            if avg_mean_return.isfinite().all():
                for _ in range(eval_interval):
                    ema.update("avg", avg_mean_return)
            avg_returns_ema.append(ema["avg"].item())

            # Save averaged model checkpoint
            if it % eval_interval == 0:
                t.save(avg_state, f"checkpoints/avg_policy_{it}.pth")

        returns.append(mean_return.item())
        if mean_return.isfinite().all():
            ema.update("cur", mean_return)
        returns_ema.append(ema["cur"].item())
        return_stds.append(std_return.item())
        gradients.append(normalization_constant.item())

        # Run validation if needed
        if val_env_factory is not None and it % val_period == 0:
            val_env = val_env_factory()
            # Evaluate current policy
            policy.eval()
            val_episode_returns = []
            for _ in range(batch_size):
                val_return = rollout_fn(policy, val_env, noise_level)
                val_episode_returns.append(val_return)
            val_mean_return = (sum(val_episode_returns) / batch_size).clamp(-100, None)
            val_returns.append(val_mean_return.item())
            if val_mean_return.isfinite().all():
                ema.update("val", val_mean_return)
            val_returns_ema.append(ema["val"].item())
            policy.train()

            # Evaluate averaged policy if using Polyak averaging
            if polyak_average:
                avg_policy = MLPPolicy(**policy_kwargs)
                avg_policy.load_state_dict(avg_state)
                avg_policy.eval()
                val_avg_episode_returns = []
                for _ in range(batch_size):
                    val_avg_return = rollout_fn(avg_policy, val_env, noise_level)
                    val_avg_episode_returns.append(val_avg_return)
                val_avg_mean_return = (sum(val_avg_episode_returns) / batch_size).clamp(
                    -100, None
                )
                val_avg_returns.append(val_avg_mean_return.item())
                if val_avg_mean_return.isfinite().all():
                    ema.update("val_avg", val_avg_mean_return)
                val_avg_returns_ema.append(ema["val_avg"].item())

        if it % eval_interval == 0:
            plot_returns(
                returns,
                returns_ema,
                return_stds,
                gradients,
                val_returns,
                val_returns_ema,
                val_period=val_period,
                eval_interval=eval_interval,
                avg_returns=avg_returns,
                avg_returns_ema=avg_returns_ema,
                val_avg_returns=val_avg_returns,
                val_avg_returns_ema=val_avg_returns_ema,
            )
            t.save(policy.state_dict(), f"checkpoints/policy_{it}.pth")

    # Return final averaged policy if using Polyak averaging, otherwise return current policy
    if polyak_average:
        final_policy = MLPPolicy(**policy_kwargs)
        final_policy.load_state_dict(avg_state)
        return final_policy
    return policy


def test_gradients():

    def helper(i: int, n: int) -> Float[TT, "n"]:
        env = PredictableEnv.create(1, 1, n + 2)
        state, _ = env.reset_state()
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
