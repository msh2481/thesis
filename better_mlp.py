import torch as t
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm


class Block(t.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = t.nn.Linear(in_features, out_features, bias=True)
        self.fc_log = t.nn.Linear(in_features, out_features, bias=False)
        self.fc_exp = t.nn.Linear(in_features, out_features, bias=False)
        t.nn.init.ones_(self.fc.weight)
        t.nn.init.zeros_(self.fc.bias)
        t.nn.init.zeros_(self.fc_log.weight)
        t.nn.init.zeros_(self.fc_exp.weight)

    def forward(self, x):
        c_exp = 5
        c_log = 0.2
        f_exp = self.fc_exp(x.clip(None, c_exp).exp() * (1 + (x - c_exp).relu()))
        f_log = self.fc_log(x.clip(c_log, None).log() - (c_log - x).relu() / c_log)
        f = self.fc(x)
        return f_exp + f_log + f


def f(x):
    return x.sqrt()


X_train = t.linspace(0, 6, 1000).unsqueeze(1)
y_train = f(X_train)
X_test = t.linspace(0, 6, 100).unsqueeze(1)
y_test = f(X_test)

model = nn.Sequential(
    Block(1, 8),
    Block(8, 1),
)
opt = t.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=0.01)

pbar = tqdm(range(10**6))
for epoch in pbar:
    p_train = model(X_train)
    eps = 0.1
    loss = ((p_train - y_train) / (y_train + eps)).square().mean()
    p_test = model(X_test)
    test_loss = t.nn.functional.mse_loss(p_test, y_test)

    opt.zero_grad()
    loss.backward()
    for p in model.parameters():
        p.grad = p.grad / (p.grad.norm() + 1e-6)
    opt.step()

    pbar.set_postfix(loss=loss.item(), test_loss=test_loss.item())
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: loss={loss.item()}, test_loss={test_loss.item()}")
        plt.plot(X_test, p_test.detach(), label="prediction")
        plt.plot(X_test, y_test, label="target")
        plt.tight_layout()
        plt.legend()
        plt.savefig(f"logs/better_mlp.png")
        plt.close()
        plt.clf()
