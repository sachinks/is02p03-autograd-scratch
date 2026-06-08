"""visualise_graph.py — render the XOR computation graph as a PNG image.

Stretch Goal 1 of IS02P03.  Uses ``torchviz.make_dot`` to walk the same
``grad_fn → next_functions`` chain that ``graph_inspect.py`` prints as text,
and renders it into a visual DAG diagram via Graphviz.

The script performs a single forward pass (no training) to build the
computation graph, then calls ``make_dot`` with named parameters so that
leaf nodes are labelled ``W1 (2, 8)``, ``b1 (8,)`` etc. instead of raw
memory addresses.

Each node in the output image shows:
  - **Leaf parameters** — labelled with name + shape, coloured light blue.
  - **Operation nodes** — labelled with the grad_fn type (e.g. AddmmBackward0).
  - **Arrows** — gradient flow direction, from loss back to weights.

Architecture matches xor_solver.py exactly:
  Input(2) -> Linear -> ReLU -> Linear -> Sigmoid -> BCE Loss
  Seed: torch.manual_seed(42), HIDDEN=8

Requires:
  pip install torchviz
  sudo apt install graphviz   (system-level, for the ``dot`` renderer)

Run:  python visualise_graph.py
Output: xor_graph.png (saved in the project directory)
"""

import torch
from torchviz import make_dot


def main() -> None:
    """Build the XOR forward graph and render it to xor_graph.png."""

    # ── Dataset ────────────────────────────────────────────────────────────
    X = torch.tensor([[0., 0.],
                      [0., 1.],
                      [1., 0.],
                      [1., 1.]])          # shape (4, 2)
    Y = torch.tensor([[0.],
                      [1.],
                      [1.],
                      [0.]])              # shape (4, 1)

    # ── Parameters (same seed + shapes as xor_solver.py) ───────────────────
    torch.manual_seed(42)
    HIDDEN = 8
    W1 = torch.randn(2, HIDDEN, requires_grad=True)   # (2, 8)
    b1 = torch.zeros(HIDDEN,    requires_grad=True)   # (8,)
    W2 = torch.randn(HIDDEN, 1, requires_grad=True)   # (8, 1)
    b2 = torch.zeros(1,         requires_grad=True)    # (1,)

    # ── Forward pass (one pass — just to build the graph) ──────────────────
    z1    = X @ W1 + b1                         # (4,8)  AddmmBackward
    a1    = torch.clamp(z1, min=0.0)            # (4,8)  ClampBackward (ReLU)
    z2    = a1 @ W2 + b2                        # (4,1)  AddmmBackward
    y_hat = 1.0 / (1.0 + torch.exp(-z2))       # (4,1)  sigmoid chain

    # ── Loss ───────────────────────────────────────────────────────────────
    eps  = 1e-7
    loss = -torch.mean(
        Y * torch.log(y_hat + eps) +
        (1 - Y) * torch.log(1 - y_hat + eps)
    )

    # ── Render ─────────────────────────────────────────────────────────────
    # params dict maps parameter tensors to display names with shapes.
    # make_dot walks grad_fn → next_functions and labels leaf nodes using
    # these names instead of hex addresses.
    params = {
        "W1 (2, 8)": W1,
        "b1 (8,)":   b1,
        "W2 (8, 1)": W2,
        "b2 (1,)":   b2,
    }

    dot = make_dot(loss, params=params, show_attrs=False, show_saved=False)
    dot.attr(rankdir="TB")          # top-to-bottom layout (loss at top)
    dot.render("xor_graph", format="png", cleanup=True)

    print("Saved: xor_graph.png")


if __name__ == "__main__":
    main()
