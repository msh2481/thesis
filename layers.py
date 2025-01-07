import math

import torch as t
from beartype import beartype as typed
from beartype.typing import Callable
from envs.diff_stock_trading_env import DiffStockTradingEnv
from jaxtyping import Float, Int, jaxtyped
from loguru import logger
from matplotlib import pyplot as plt
from optimizers import CautiousAdamW, CautiousLion, CautiousSGD
from torch import nn, Tensor as TT
from tqdm.auto import tqdm


class DimWiseLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dim: int,
        bias: bool = True,
        gain: float = 1.0,
    ):
        """Linear layer that operates along a specific dimension.

        Args:
            in_features: Size of input features along dim
            out_features: Size of output features along dim
            dim: Dimension (not counting batch dim) to apply linear transform (0 or 1)
            bias: Whether to include bias term (default: True)
        """
        super().__init__()
        assert dim in [0, 1], "Dim must be 0 or 1"
        self.dim = dim
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.gain = gain
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights using orthogonal initialization."""
        nn.init.orthogonal_(self.linear.weight)
        self.linear.weight.data *= self.gain
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    @typed
    def forward(
        self,
        x: Float[TT, "batch n m"],
    ) -> Float[TT, "batch out m"] | Float[TT, "batch n out"]:
        x = x.movedim(1 + self.dim, -1)
        x = self.linear(x)
        x = x.movedim(-1, 1 + self.dim)
        return x


class DimWisePair(nn.Module):
    @typed
    def __init__(self, n_in: int, m_in: int, n_out: int, m_out: int):
        super().__init__()
        self.n_wise = DimWiseLinear(n_in, n_out, dim=0)
        self.m_wise = DimWiseLinear(m_in, m_out, dim=1)

    @typed
    def forward(
        self, x: Float[TT, "batch n_in m_in"]
    ) -> Float[TT, "batch n_out m_out"]:
        x = self.n_wise(x)
        x = self.m_wise(x)
        return x


class PermInvariantFeatures(nn.Module):
    def __init__(self):
        """Combination of certain permutation-invariant functions, along `dim=0` (not counting batch)."""
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=False)
        self.linear.weight.data[0, 0] = 1.0
        self.linear.weight.data[0, 1:] = 0.0

    @jaxtyped(typechecker=typed)
    def forward(
        self, x: Float[TT, "batch stock features"]
    ) -> Float[TT, "batch stock features"]:
        mn = x.min(dim=1, keepdim=True).values
        mx = x.max(dim=1, keepdim=True).values
        eps = 1e-8
        # Base functions
        softmax = t.softmax(x, dim=1)
        uniform = (x - mn) / (mx - mn + eps)
        softmin = t.softmax(-x, dim=1)
        # features = t.stack([x, softmin, uniform, softmax], dim=-1)
        features = t.stack([x, softmax], dim=-1)
        return self.linear(features).squeeze(-1)


class SortedDimWiseLinear(nn.Module):
    def __init__(self, n_features: int, dim: int, bias: bool = False):
        """Permutation-invariant kinda-linear layer that operates along a specific dimension.

        Args:
            n_features: Number of input/output features along dim
            dim: Dimension (not counting batch dim) to apply linear transform (0 or 1)
            bias: Whether to include bias term (default: True)
        """
        super().__init__()
        assert dim in [0, 1], "Dim must be 0 or 1"
        self.dim = dim
        self.linear = nn.Linear(n_features, n_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights using orthogonal initialization."""
        nn.init.eye_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    @typed
    def forward(
        self,
        x: Float[TT, "batch n m"],
    ) -> Float[TT, "batch n m"]:
        x = x.movedim(1 + self.dim, -1)
        argsort = x.argsort(dim=-1)
        inverse = t.zeros_like(argsort)
        inverse.scatter_(-1, argsort, t.arange(x.shape[-1]).repeat((*x.shape[:-1], 1)))
        x = x.scatter(-1, argsort, x)
        x = self.linear(x)
        x = x.scatter(-1, inverse, x)
        x = x.movedim(-1, 1 + self.dim)
        return x


class DimWiseMLP(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, dim: int, hidden: int | None = None
    ):
        super().__init__()
        assert dim in [0, 1], "Dim must be 0 or 1"
        self.dim = dim
        self.in_features = in_features
        self.out_features = out_features
        self.hidden = hidden or 4 * in_features
        self.first = nn.Linear(self.in_features, self.hidden)
        self.second = nn.Linear(self.hidden, self.out_features)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights using orthogonal initialization."""
        nn.init.orthogonal_(self.first.weight)
        nn.init.normal_(self.second.weight, std=0.01)
        if self.first.bias is not None:
            nn.init.zeros_(self.first.bias)
        if self.second.bias is not None:
            nn.init.zeros_(self.second.bias)

    @typed
    def forward(
        self,
        x: Float[TT, "batch n in_features"],
    ) -> Float[TT, "batch n out_features"]:
        x = x.movedim(1 + self.dim, -1)
        skip = x
        x = self.first(x)
        x = t.relu(x)
        x = self.second(x)
        x = x.movedim(-1, 1 + self.dim)
        return x


class SortedDimWiseMLP(nn.Module):
    def __init__(self, n_features: int, dim: int, hidden: int | None = None):
        """Permutation-invariant kinda-linear layer that operates along a specific dimension.

        Args:
            n_features: Number of input/output features along dim
            dim: Dimension (not counting batch dim) to apply linear transform (0 or 1)
            bias: Whether to include bias term (default: True)
        """
        super().__init__()
        assert dim in [0, 1], "Dim must be 0 or 1"
        self.dim = dim
        self.n_features = n_features
        self.hidden = hidden or 4 * n_features
        self.first = nn.Linear(self.n_features, self.hidden)
        self.second = nn.Linear(self.hidden, self.n_features)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights using orthogonal initialization."""
        nn.init.orthogonal_(self.first.weight)
        nn.init.normal_(self.second.weight, std=0.01)
        if self.first.bias is not None:
            nn.init.zeros_(self.first.bias)
        if self.second.bias is not None:
            nn.init.zeros_(self.second.bias)

    @typed
    def forward(
        self,
        x: Float[TT, "batch n m"],
    ) -> Float[TT, "batch n m"]:
        x = x.movedim(1 + self.dim, -1)
        argsort = x.argsort(dim=-1)
        inverse = t.zeros_like(argsort)
        inverse.scatter_(-1, argsort, t.arange(x.shape[-1]).repeat((*x.shape[:-1], 1)))
        x = x.scatter(-1, argsort, x)
        skip = x
        x = self.first(x)
        x = t.relu(x)
        x = self.second(x) + skip
        x = x.scatter(-1, inverse, x)
        x = x.movedim(-1, 1 + self.dim)
        return x


class StockBlock(nn.Module):
    @typed
    def __init__(
        self, n_stock: int, in_features: int, out_features: int, residual: bool = False
    ) -> None:
        super().__init__()
        self.sorted_mlp = SortedDimWiseMLP(n_stock, dim=0, hidden=4 * n_stock)
        self.feature_wise = DimWiseMLP(
            in_features, out_features, dim=1, hidden=4 * in_features
        )
        self.residual = residual
        if residual:
            assert in_features == out_features

    def forward(
        self, x: Float[TT, "batch n_stock in_features"]
    ) -> Float[TT, "batch n_stock out_features"]:
        y = self.sorted_mlp(x)
        y = self.feature_wise(y)
        if self.residual:
            y = x + y
        return y


class Residual(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: Float[TT, "..."]) -> Float[TT, "..."]:
        return x + self.module(x)


def dimwise_test_fn(x: Float[TT, "batch 4 5"]) -> Float[TT, "batch 4 3"]:
    batch = x.shape[0]
    y = t.zeros((batch, 4, 3))
    # y[:, :, 0] = x[:, :, 0]
    # y[:, :, 1] = x[:, :, 1] + x[:, :, 2]
    y[:, :, 2] = t.maximum(x[:, :, 3], x[:, :, 4])
    z = y.softmax(dim=-1)
    return y


def test_stockblock():
    model = nn.Sequential(
        StockBlock(4, 5, 5, True),
        DimWiseLinear(5, 3, 1),
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.warning(f"#params = {n_params}")
    # init all layers inside residuals to small values
    for i in range(1, len(model) - 1):
        for name, param in model[i].named_parameters():
            if "weight" in name:
                param.data.uniform_(-1e-3, 1e-3)

    opt = t.optim.Adam(model.parameters(), lr=1e-4)

    n_batches = 100000
    batch_size = 1000
    pbar = tqdm(range(n_batches))

    for i in pbar:
        x = t.randn(batch_size, 4, 5)
        y_true = dimwise_test_fn(x)

        y_pred = model(x)
        loss = nn.functional.mse_loss(y_pred, y_true)

        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % 100 == 0:
            pbar.set_postfix(loss=f"{loss.item():.5f}")
        if i % 1000 == 0:
            with t.no_grad():
                logger.info(f"Batch {i}, Loss: {loss.item():.3e}")
                if i % 2000 == 0:
                    x_test = t.randn(1, 4, 5)
                    y_true = dimwise_test_fn(x_test)
                    y_pred = model(x_test)
                    logger.info(f"Test input:\n{x_test[0]}")
                    logger.info(f"True output:\n{y_true[0]}")
                    logger.info(f"Pred output:\n{y_pred[0]}")


def test_perm_invariant():
    batch = 2
    stock = 3
    features = 2
    x = t.randn((batch, stock, features))
    # f = StockBlock(n_stock=stock, in_features=features, out_features=4)
    f = PermInvariantFeatures()
    ys = f(x)
    y = ys[0]
    logger.debug(f"input:\n{x[0].numpy()}")
    logger.debug(f"shape: {y.shape}")
    logger.debug(f"values:\n{y.detach().numpy()}")
    y = ys[1]
    logger.debug(f"values:\n{y.detach().numpy()}")


def test_sorted():
    batch = 2
    stock = 3
    features = 2
    x = t.randn((batch, stock, features))
    # f = StockBlock(n_stock=stock, in_features=features, out_features=4)
    f = SortedDimWiseLinear(stock, dim=0)
    ys = f(x)
    y = ys[0]
    logger.debug(f"input:\n{x[0].numpy()}")
    logger.debug(f"shape: {y.shape}")
    logger.debug(f"values:\n{y.detach().numpy()}")
    y = ys[1]
    logger.debug(f"values:\n{y.detach().numpy()}")


if __name__ == "__main__":
    test_stockblock()
    # test_perm_invariant()
    # test_sorted()
