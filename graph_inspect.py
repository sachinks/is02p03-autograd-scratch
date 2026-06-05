"""graph_inspect.py — make the computation graph visible.

PyTorch's computation graph is not a separate object you can ask for.
It lives inside the tensors themselves: each non-leaf tensor has a
``.grad_fn`` (the C++ Function object that created it and knows how to
differentiate it), and each ``grad_fn`` has ``.next_functions`` pointing
at the ``grad_fn``s of its inputs.  Following that chain *is* walking the
graph.

This file:
  1. Runs a single forward pass of a 2-8-1 XOR network to build the graph.
  2. Prints the ``.grad_fn`` of each intermediate tensor.
  3. Recursively walks the graph via ``.next_functions`` from the output
     back to the four ``AccumulateGrad`` leaf nodes (W1, b1, W2, b2).

Uses the same architecture as xor_solver.py (HIDDEN=8) for consistent
weight shapes across the project.

Run:  python graph_inspect.py
"""

import torch

torch.manual_seed(42)
HIDDEN = 8   # matches xor_solver.py — consistent weight shapes across project

x  = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
W1 = torch.randn(2, HIDDEN, requires_grad=True)   # (2, 8)
b1 = torch.zeros(HIDDEN,    requires_grad=True)   # (8,)
W2 = torch.randn(HIDDEN, 1, requires_grad=True)   # (8, 1)
b2 = torch.zeros(1,         requires_grad=True)   # (1,)

# Forward pass — each line allocates one grad_fn node.
z1     = x @ W1 + b1                   # MmBackward / AddBackward
a1     = torch.relu(z1)                # ReluBackward
z2     = a1 @ W2 + b2                  # MmBackward / AddBackward
output = torch.sigmoid(z2)            # SigmoidBackward

print("=== grad_fn of each tensor ===")
print(f"x.grad_fn      : {x.grad_fn}   <- leaf, no op created it")
print(f"W1.grad_fn     : {W1.grad_fn}  <- leaf parameter (2x{HIDDEN})")
print(f"z1.grad_fn     : {z1.grad_fn}")
print(f"a1.grad_fn     : {a1.grad_fn}")
print(f"z2.grad_fn     : {z2.grad_fn}")
print(f"output.grad_fn : {output.grad_fn}")


def walk(fn, depth: int = 0) -> None:
    """Recursively print the grad_fn DAG from *fn* back to the leaf nodes.

    At each node, prints the ``grad_fn`` class name indented by *depth*
    levels, then recurses into ``fn.next_functions`` — the list of
    (parent_fn, output_index) pairs that point further toward the inputs.

    ``AccumulateGrad`` nodes at the bottom are the leaf parameters; that is
    where ``.backward()`` deposits the final gradients into ``.grad``.

    Args:
        fn: a ``grad_fn`` object (e.g. ``SigmoidBackward``) or ``None``
            (for leaf tensors — recursion base case).
        depth: current recursion depth, used only for indentation.
    """
    if fn is None:
        return
    print("    " * depth + f"-> {type(fn).__name__}")
    for parent, _ in getattr(fn, "next_functions", ()):
        walk(parent, depth + 1)


print("\n=== computation graph (output -> leaves) ===")
walk(output.grad_fn)
print("\nAccumulateGrad nodes are the leaf parameters (W1, b1, W2, b2):")
print("that is where .backward() deposits the final gradients into .grad.")
print(f"\nWeight shapes: W1={list(W1.shape)}, b1={list(b1.shape)}, "
      f"W2={list(W2.shape)}, b2={list(b2.shape)}")
