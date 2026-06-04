"""
xor_solver.py - 2-layer XOR network using RAW PyTorch tensors only.

No nn.Module, no nn.Linear, no F.relu, no torch.sigmoid, no optimiser.
Every operation in the forward pass, backward pass, weight update and
gradient reset is written explicitly so the autograd mechanism is visible.

Run:  python xor_solver.py
"""

import torch

# ----------------------------------------------------------------------
# DATASET
# XOR: output is 1 when the two inputs differ, 0 when they are the same.
# This is the classic problem a SINGLE linear layer cannot solve - it is
# not linearly separable - so we need a hidden layer + non-linearity.
# ----------------------------------------------------------------------
X = torch.tensor([[0., 0.],
                  [0., 1.],
                  [1., 0.],
                  [1., 1.]])          # shape (4, 2)
Y = torch.tensor([[0.],
                  [1.],
                  [1.],
                  [0.]])              # shape (4, 1)

# ----------------------------------------------------------------------
# PARAMETERS
# requires_grad=True is the switch that turns on graph tracking. From now
# on, every operation these tensors take part in records a node so that
# .backward() can later compute d(loss)/d(parameter).
# manual_seed makes the random init reproducible - same run every time.
# ----------------------------------------------------------------------
torch.manual_seed(42)
HIDDEN = 8   # 8, not 4: a 4-neuron ReLU net solves XOR only ~60% of seeds
             # (dead-ReLU local minima). 8 units solve it on every seed.
W1 = torch.randn(2, HIDDEN, requires_grad=True)   # input(2) -> hidden(8)
b1 = torch.zeros(HIDDEN,    requires_grad=True)
W2 = torch.randn(HIDDEN, 1, requires_grad=True)   # hidden(8) -> output(1)
b2 = torch.zeros(1,         requires_grad=True)

params = [W1, b1, W2, b2]
lr = 0.1
EPOCHS = 2000


def relu(x):
    """ReLU = max(0, x). Local gradient is 1 where x > 0, else 0."""
    return torch.clamp(x, min=0.0)


def sigmoid(x):
    """Sigmoid = 1 / (1 + e^-x). Squashes any real number into (0, 1)."""
    return 1.0 / (1.0 + torch.exp(-x))


def forward(x):
    """x -> Linear -> ReLU -> Linear -> Sigmoid. Returns probability (4,1)."""
    z1 = x @ W1 + b1          # (4,2)@(2,8) + (8,) -> (4,8)  bias broadcasts
    a1 = relu(z1)             # (4,8) element-wise
    z2 = a1 @ W2 + b2         # (4,8)@(8,1) + (1,) -> (4,1)
    return sigmoid(z2)        # (4,1) probabilities


def bce_loss(y_hat, y_true):
    """Binary cross-entropy: -mean[ y*log(p) + (1-y)*log(1-p) ].
    eps keeps log() away from log(0) = -inf (a source of NaN loss)."""
    eps = 1e-7
    return -torch.mean(
        y_true * torch.log(y_hat + eps) +
        (1 - y_true) * torch.log(1 - y_hat + eps)
    )


def train():
    for epoch in range(EPOCHS):
        # ---- FORWARD: build the graph fresh this epoch ----------------
        y_hat = forward(X)
        loss = bce_loss(y_hat, Y)

        # ---- BACKWARD: walk the graph in reverse, fill .grad ----------
        loss.backward()       # populates W1.grad, b1.grad, W2.grad, b2.grad

        # ---- UPDATE: gradient descent step, OUTSIDE the graph ---------
        # torch.no_grad() stops the subtraction from being recorded as new
        # graph nodes (the update is not part of the model's computation).
        with torch.no_grad():
            for p in params:
                p -= lr * p.grad

        # ---- ZERO GRADS: mandatory, .grad ACCUMULATES otherwise -------
        for p in params:
            p.grad.zero_()

        if epoch % 200 == 0:
            print(f"epoch {epoch:4d}  loss={loss.item():.4f}")

    return loss.item()


def evaluate():
    """Print final predictions. No graph needed -> no_grad saves memory."""
    with torch.no_grad():
        preds = forward(X)
    print("\nFinal predictions (probability that output == 1):")
    for inp, p, target in zip(X, preds, Y):
        rounded = int(p.item() > 0.5)
        ok = "OK" if rounded == int(target.item()) else "WRONG"
        print(f"  {inp.tolist()} -> {p.item():.3f}  (rounds to {rounded}, "
              f"target {int(target.item())})  {ok}")


if __name__ == "__main__":
    final_loss = train()
    print(f"\nfinal loss = {final_loss:.4f}")
    evaluate()
