import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch as t
from agent import sample_agent
from beartype import beartype as typed
from jaxtyping import Float
from loguru import logger
from matplotlib import pyplot as plt
from numpy import ndarray as ND
from numpy.random import default_rng
from params import some_insight_params
from price_process import simulate, synthetic_data
from torch import Tensor as TT
from tqdm.auto import tqdm
from visualization import visualize_agents, visualize_trades


@typed
def collect_data(n_series: int) -> Float[TT, "series steps 6"]:
    results = []
    for i in tqdm(range(n_series)):
        params = {
            "alpha_0": np.random.normal(1.0, 0.2),
            "alpha_1": np.random.normal(1.0, 0.2),
            "k_alpha": np.random.lognormal(0, 0.5),
            "k_theta": np.random.lognormal(7, 1),
            "sigma_alpha": np.random.lognormal(0, 0.5),
            "sigma_theta": np.random.lognormal(-2, 0.5),
            "w_m": np.random.lognormal(7, 1),
            "w_alpha": np.random.lognormal(0.4, 0.3),
            "mu_D": np.random.normal(0, 0.2),
            "sigma_D": np.random.lognormal(0, 0.5),
            "mu_A": np.random.normal(0, 0.2),
            "sigma_A": np.random.lognormal(0, 0.5),
            "delta_alpha": np.random.lognormal(0, 0.5),
            "delta_theta": np.random.lognormal(-4.6, 0.5),
            "lambda": np.random.lognormal(1.6, 0.5),
            "price_scale": np.random.lognormal(-7, 1),
            "use_levy": True,
            "levy_param": np.random.normal(1.4, 0.1),
            "initial_price": np.random.lognormal(4.6, 0.5),
        }
        """ 
        candle_data[i, 0] = candle.open
        candle_data[i, 1] = candle.high
        candle_data[i, 2] = candle.low
        candle_data[i, 3] = candle.close
        candle_data[i, 4] = candle.volume
        """
        current = synthetic_data(params, steps=1000, period=5, seed=i)
        results.append(current)
    return t.tensor(np.stack(results))


def generate():
    k = 10
    raw_data = collect_data(n_series=3000)
    samples = []
    targets = []
    for i_series in tqdm(range(raw_data.shape[0])):
        for i_step in range(k, raw_data.shape[1], k):
            samples.append(raw_data[i_series, i_step - k : i_step, :5].flatten())
            targets.append(raw_data[i_series, i_step - 1, 5])
    samples = t.stack(samples)
    targets = t.stack(targets)
    # Save samples and targets to files
    t.save(samples, "data/samples.pt")
    t.save(targets, "data/targets.pt")
    print(samples.shape, targets.shape)


k = 10


class MLPBlock(t.nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.features = features
        self.linear = t.nn.Linear(features, features)
        self.act = t.nn.Mish()

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.act(self.linear(x)) + x


class AtLeast1(t.nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return t.relu(x - 1) + 1


def build_model():
    b = 200
    b2 = 32
    model = t.nn.Sequential(
        t.nn.Linear(5 * k, b),
        MLPBlock(b),
        MLPBlock(b),
        t.nn.Linear(b, b2),
        MLPBlock(b2),
        MLPBlock(b2),
        MLPBlock(b2),
        MLPBlock(b2),
        MLPBlock(b2),
        MLPBlock(b2),
        t.nn.Linear(b2, 1),
        AtLeast1(),
    )
    for m in model.modules():
        if isinstance(m, t.nn.Linear):
            t.nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                t.nn.init.zeros_(m.bias)
    return model


def criterion(preds: t.Tensor, targets: t.Tensor) -> t.Tensor:
    assert (targets > 0).all()
    assert (preds >= 0).all()
    assert (preds > 0).all()
    return t.log(preds / targets).square().sum()


def train():
    model = build_model()
    optimizer = t.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    samples = t.load("data/samples.pt").float()
    targets = t.load("data/targets.pt").float()

    # Add validation split
    n_samples = samples.shape[0]
    print("N = ", n_samples)
    n_train = int(0.8 * n_samples)
    n_val = n_samples - n_train
    indices = t.randperm(n_samples)

    train_samples = samples[indices[:n_train]]
    train_targets = targets[indices[:n_train]]
    val_samples = samples[indices[n_train:]]
    val_targets = targets[indices[n_train:]]

    print(f"Training samples: {train_samples.shape[0]}")
    print(f"Validation samples: {val_samples.shape[0]}")

    best_val_loss = float("inf")
    patience = 100
    patience_counter = 0

    # Lists to store losses
    train_losses = []
    val_losses = []
    reg_losses = []

    for i in tqdm(range(100000)):
        # Training
        model.train()
        preds = model(train_samples).flatten()
        assert preds.shape == train_targets.shape
        mse = criterion(preds, train_targets) / n_train
        loss = mse

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with t.no_grad():
            val_preds = model(val_samples).flatten()
            val_mse = criterion(val_preds, val_targets) / n_val
            val_loss = val_mse

        # Store losses
        train_losses.append(mse.item())
        val_losses.append(val_mse.item())

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            t.save(model.state_dict(), "models/averaging_model_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at iteration {i}")
                break

        if i & (i - 1) == 0:
            print(
                f"Iteration {i:4d} | Train MSE: {mse.item():.6f} | Val MSE: {val_mse.item():.6f}"
            )

    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label="Training MSE", alpha=0.7)
    plt.plot(val_losses, label="Validation MSE", alpha=0.7)
    plt.plot(reg_losses, label="Regularization", alpha=0.7)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Loss (log scale)")
    plt.title("Training History")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("img/training_history.png")
    plt.close()

    # Save training history
    t.save(
        {
            "train_losses": t.tensor(train_losses),
            "val_losses": t.tensor(val_losses),
            "reg_losses": t.tensor(reg_losses),
        },
        "models/training_history.pt",
    )

    # Load best model and continue with visualization...
    model.load_state_dict(t.load("models/averaging_model_best.pt"))

    # Rest of the visualization code remains the same...
    # weights = model.weight.data.reshape(k, 5)
    # print("\nModel weights (k x 5):")
    # print("-" * 50)
    # print(f"{'Open':>10} {'High':>10} {'Low':>10} {'Close':>10} {'Volume':>10}")
    # print("-" * 50)
    # for row in weights:
    #     print("".join(f"{x:10.3f}" for x in row))
    # print("-" * 50)
    # print("\nModel bias:")
    # if model.bias is not None:
    #     print(f"{model.bias.data.item():.3f}")
    # else:
    #     print("No bias")

    # # Save final model state dict
    # t.save(model.state_dict(), "models/averaging_model.pt")
    # print("\nModel saved to models/averaging_model.pt")

    # plt.figure(figsize=(10, 8))
    # sns.heatmap(
    #     weights.numpy(),
    #     xticklabels=["Open", "High", "Low", "Close", "Volume"],
    #     yticklabels=[f"t-{i}" for i in range(weights.shape[0] - 1, -1, -1)],
    #     cmap="RdBu",
    #     center=0,
    #     annot=True,
    #     fmt=".3f",
    # )
    # plt.title("Model Weights Heatmap")
    # plt.tight_layout()
    # plt.savefig("img/weights_heatmap.png")
    # plt.close()


def predict():
    model = build_model()
    model.load_state_dict(t.load("models/averaging_model_best.pt"))
    # model.weight.data[:, :] = 0
    # model.weight.data[-1, :4] = 0.25
    data = collect_data(n_series=10)
    for series in data:
        features = []
        targets = []
        close = []
        for i in range(k, len(series)):
            features.append(t.tensor(series[i - k : i, :5].float().flatten()))
            targets.append(series[i - 1, 5])
            close.append(series[i - 1, 3])
        print(len(features))
        features = t.stack(features)
        targets = t.tensor(targets)
        close = t.tensor(close)
        preds = model(features).flatten().detach()
        mse = t.nn.functional.mse_loss(preds, targets, reduction="sum").item()
        mse0 = t.nn.functional.mse_loss(close, targets, reduction="sum").item()
        print(f"MSE: {mse:.6f} | MSE0: {mse0:.6f}")
        plt.plot(preds, label="average")
        plt.plot(targets, label="fair")
        plt.plot(close, label="close")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # generate()
    train()
    # predict()
    # data = collect_data(n_series=10)
    # for series in data:
    #     plt.plot(series[:, 3], label="close")
    #     plt.plot(series[:, 5], label="fair")
    #     plt.legend()
    #     plt.show()
