"""
graph_inspect.py - make the computation graph visible.

The graph is NOT a separate object you can ask PyTorch for. It lives inside
the tensors: each non-leaf tensor has a .grad_fn (the backward function that
made it), and each grad_fn has .next_functions pointing at the grad_fns of
its inputs. Following that chain IS walking the graph.

Run:  python graph_inspect.py
"""

import torch

torch.manual_seed(42)
x  = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
W1 = torch.randn(2, 4, requires_grad=True)
b1 = torch.zeros(4,    requires_grad=True)
W2 = torch.randn(4, 1, requires_grad=True)
b2 = torch.zeros(1,    requires_grad=True)

# Forward pass - each line allocates one grad_fn node.
z1     = x @ W1 + b1                   # AddmmBackward
a1     = torch.relu(z1)                # ReluBackward
z2     = a1 @ W2 + b2                  # AddmmBackward
output = torch.sigmoid(z2)            # SigmoidBackward

print("=== grad_fn of each tensor ===")
print("x.grad_fn      :", x.grad_fn,      "  <- leaf, no op created it")
print("W1.grad_fn     :", W1.grad_fn,     "  <- leaf parameter")
print("z1.grad_fn     :", z1.grad_fn)
print("a1.grad_fn     :", a1.grad_fn)
print("z2.grad_fn     :", z2.grad_fn)
print("output.grad_fn :", output.grad_fn)


def walk(fn, depth=0):
    """Recursively print the grad_fn DAG from output back to the leaves."""
    if fn is None:
        return
    print("    " * depth + f"-> {type(fn).__name__}")
    for parent, _ in getattr(fn, "next_functions", ()):
        walk(parent, depth + 1)


print("\n=== computation graph (output -> leaves) ===")
walk(output.grad_fn)
print("\nAccumulateGrad nodes are the leaf parameters (W1, b1, W2, b2):")
print("that is where .backward() deposits the final gradients into .grad.")
