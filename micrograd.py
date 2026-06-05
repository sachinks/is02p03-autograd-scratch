"""micrograd.py — a tiny autograd engine built from scratch (no torch).

Re-implements, in ~200 lines of plain Python, exactly what PyTorch's
autograd does internally:

  Value     one node in the computation graph: wraps a scalar + its gradient
            + a _backward closure + references to parent nodes.
  backward  topological sort of the graph, then chain-rule in reverse order.
  MLP       a 2-8-1 feedforward network composed from Value nodes.

Solving XOR with this engine proves the mechanism xor_solver.py gets "for
free" from PyTorch is not magic — it is ~15 lines of Python once you
understand what grad_fn and next_functions actually do.

Key differences from xor_solver.py:
  - Scalar-by-scalar arithmetic (no vectorised tensors) — much slower.
  - lr=0.5, 3000 epochs (vs lr=0.1, 2000) because the scalar engine
    converges more slowly without vectorised batch updates.
  - random.seed(42) instead of torch.manual_seed(42).
  - Architecture is identical: 2-8-1, ReLU hidden, Sigmoid output, BCE loss.

Run:  python micrograd.py   (no dependencies beyond the standard library)
"""

import math
import random


class Value:
    """A single scalar value and its gradient in the autograd computation graph.

    Each ``Value`` object is one node in a dynamically-built DAG.
    Arithmetic operators (+, *, **, relu, sigmoid, log) each return a **new**
    ``Value`` node and set its ``_backward`` closure to the local gradient
    formula for that operation.  Calling ``.backward()`` on the final loss
    node traverses the DAG in reverse topological order and calls every
    ``_backward`` closure, propagating gradients back to the leaf nodes
    (the network parameters).

    This mirrors exactly what PyTorch's C++ engine does, just slower and
    in Python so it can be read.

    Attributes:
        data (float): the scalar forward value of this node.
        grad (float): accumulated gradient ``d(root)/d(self)``.
            Starts at 0.0; populated by ``.backward()``.
        _backward (callable): closure that pushes gradient from this node
            to its parent nodes.  Set by each operator.  Default is no-op.
        _prev (set[Value]): parent nodes — the Values this node was computed
            from.  Used by ``.backward()`` to build the topological order.
        _op (str): string label of the operation that created this node
            (e.g. ``"+"``, ``"relu"``).  Used only for debugging.
    """

    def __init__(self, data: float, _children: tuple = (), _op: str = ""):
        """Create a leaf Value node (no parent, no backward formula).

        Args:
            data: the scalar value.
            _children: parent Values this node was computed from.  Passed
                by operators; callers should not set this directly.
            _op: operation label for debugging.  Passed by operators.
        """
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other: "Value | float") -> "Value":
        """Add two Values: ``out = self + other``.

        Local gradient: ``d(out)/d(self) = 1``, ``d(out)/d(other) = 1``.
        Gradients just pass through unchanged.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            """Distribute out.grad to both addends unchanged (d(a+b)/da = d(a+b)/db = 1)."""
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other: "Value | float") -> "Value":
        """Multiply two Values: ``out = self * other``.

        Local gradient (product rule):
          ``d(out)/d(self) = other.data``
          ``d(out)/d(other) = self.data``
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            """Product rule: d(a*b)/da = b.data, d(a*b)/db = a.data."""
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, p: int | float) -> "Value":
        """Raise self to a constant power: ``out = self ** p``.

        Local gradient (power rule): ``d(out)/d(self) = p * self.data^(p-1)``.

        Args:
            p: a numeric constant (int or float).  Not a Value — exponent
               must be a constant for the gradient formula to hold.
        """
        assert isinstance(p, (int, float))
        out = Value(self.data ** p, (self,), f"**{p}")

        def _backward():
            """Power rule: d(x^p)/dx = p * x^(p-1)."""
            self.grad += (p * self.data ** (p - 1)) * out.grad
        out._backward = _backward
        return out

    def relu(self) -> "Value":
        """Apply ReLU: ``out = max(0, self.data)``.

        Local gradient: 1 where input > 0, else 0.
        The gradient is blocked (zeroed) for negative inputs — this is
        the mechanism behind dead neurons when all inputs to a neuron are
        negative for every training example.
        """
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():
            """Pass gradient only where input > 0; zero elsewhere (dead-ReLU region)."""
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self) -> "Value":
        """Apply sigmoid: ``out = 1 / (1 + exp(-self.data))``.

        Local gradient: ``s * (1 - s)`` where ``s = sigmoid(self.data)``.
        Maximum value is 0.25 (at s=0.5) — this is why sigmoid causes
        vanishing gradients in deep networks.
        """
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, (self,), "sigmoid")

        def _backward():
            """Sigmoid local gradient: s*(1-s), maximum 0.25 at s=0.5."""
            self.grad += (s * (1 - s)) * out.grad
        out._backward = _backward
        return out

    def log(self) -> "Value":
        """Apply natural logarithm: ``out = ln(self.data)``.

        Local gradient: ``1 / self.data``.
        Used in the BCE loss.  Will produce ``-inf`` if ``self.data <= 0``
        — callers should add an epsilon before calling ``.log()``.
        """
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            """Log local gradient: 1/x. Caller must ensure self.data > 0."""
            """Log local gradient: 1/x."""
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    def backward(self) -> None:
        """Run reverse-mode autodiff from this node back to all leaf nodes.

        Builds a topological ordering of all ``Value`` nodes reachable via
        ``._prev`` (DFS, visited set), then iterates in reverse order —
        outputs before inputs — calling ``._backward()`` on each node.

        Sets ``self.grad = 1.0`` as the seed (``d(self)/d(self) = 1``).

        After this call, every leaf node's ``.grad`` holds the gradient of
        this node with respect to that leaf, i.e. what PyTorch deposits into
        ``.grad`` when you call ``loss.backward()``.
        """
        topo, visited = [], set()

        def build(v):
            """DFS post-order traversal: append v after all its children."""
            if v not in visited:
                visited.add(v)
                for parent in v._prev:
                    build(parent)
                topo.append(v)
        build(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    # ── helpers ──────────────────────────────────────────────────────────
    def __neg__(self) -> "Value":
        """Negate: ``-self``."""
        return self * -1

    def __radd__(self, o) -> "Value":
        """Right-hand add: ``o + self`` (e.g. ``sum(..., start)`` pattern)."""
        return self + o

    def __sub__(self, o) -> "Value":
        """Subtract: ``self - o``."""
        return self + (-o)

    def __rsub__(self, o) -> "Value":
        """Right-hand subtract: ``o - self``."""
        return (-self) + o

    def __rmul__(self, o) -> "Value":
        """Right-hand multiply: ``o * self``."""
        return self * o

    def __truediv__(self, o: "Value | float") -> "Value":
        """Divide: ``self / o`` implemented as ``self * o^(-1)``."""
        o = o if isinstance(o, Value) else Value(o)
        return self * o ** -1

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


# ── Network layers ─────────────────────────────────────────────────────────

class Neuron:
    """A single artificial neuron: weighted sum of inputs + bias, then activation.

    Attributes:
        w (list[Value]): weight for each input dimension.
        b (Value): bias term.
        activation (str): ``"relu"`` or ``"sigmoid"``.
    """

    def __init__(self, n_in: int, activation: str):
        """Initialise weights from Uniform(-1, 1) and bias to 0.

        Args:
            n_in: number of input dimensions (number of weights).
            activation: ``"relu"`` or ``"sigmoid"``.
        """
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0.0)
        self.activation = activation

    def __call__(self, xs: list) -> Value:
        """Compute ``activation(w·x + b)`` for input list *xs*.

        Args:
            xs: list of ``Value`` objects or raw floats, length == n_in.

        Returns:
            A ``Value`` node holding the neuron's output.
        """
        act = sum((wi * xi for wi, xi in zip(self.w, xs)), self.b)
        return act.relu() if self.activation == "relu" else act.sigmoid()

    def parameters(self) -> list:
        """Return all trainable parameters (weights + bias) as a flat list."""
        return self.w + [self.b]


class Layer:
    """A fully-connected layer: a list of Neurons applied in parallel.

    Attributes:
        neurons (list[Neuron]): one Neuron per output unit.
    """

    def __init__(self, n_in: int, n_out: int, activation: str):
        """Create *n_out* neurons each with *n_in* inputs.

        Args:
            n_in: number of input features.
            n_out: number of output neurons (layer width).
            activation: activation function passed to each Neuron.
        """
        self.neurons = [Neuron(n_in, activation) for _ in range(n_out)]

    def __call__(self, xs: list) -> list:
        """Apply every neuron to the same input *xs*.

        Args:
            xs: list of input Values or floats, length == n_in.

        Returns:
            List of Value outputs, one per neuron.
        """
        return [n(xs) for n in self.neurons]

    def parameters(self) -> list:
        """Return all parameters from all neurons as a flat list."""
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    """A 2-8-1 multi-layer perceptron for XOR classification.

    Architecture:
        Layer 1: 2 inputs -> 8 hidden units, ReLU activation
        Layer 2: 8 hidden units -> 1 output unit, Sigmoid activation

    8 hidden units (not 4) to avoid dead-ReLU local minima — see module
    docstring for details.
    """

    def __init__(self):
        """Initialise the two layers with random weights."""
        self.l1 = Layer(2, 8, "relu")
        self.l2 = Layer(8, 1, "sigmoid")

    def __call__(self, xs: list) -> Value:
        """Run a forward pass through both layers.

        Args:
            xs: list of 2 input Values or floats.

        Returns:
            A single Value node holding the output probability.
        """
        return self.l2(self.l1(xs))[0]

    def parameters(self) -> list:
        """Return all parameters from both layers as a flat list."""
        return self.l1.parameters() + self.l2.parameters()


def main() -> None:
    """Train the from-scratch MLP on XOR and print final predictions.

    Training loop (structurally identical to xor_solver.py):
      1. Forward pass — compute loss over all 4 XOR examples.
      2. Zero gradients — reset every parameter's ``.grad`` to 0.0.
         (Values accumulate gradients just like PyTorch tensors do.)
      3. Backward pass — call ``loss.backward()`` to propagate gradients.
      4. SGD update  — subtract ``lr * param.grad`` from each param.

    Hyperparameters differ from xor_solver.py:
      - lr=0.5 (higher, compensates for slower scalar-by-scalar convergence)
      - 3000 epochs (more steps needed without vectorised batch updates)

    Prints loss every 500 epochs, then the 4 final predictions.
    """
    random.seed(42)
    data = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]
    model = MLP()
    lr = 0.5

    for epoch in range(3000):
        loss = Value(0.0)
        eps = 1e-7
        for xs, y in data:
            p = model([Value(x) for x in xs])
            loss = loss + -(Value(y) * (p + eps).log()
                            + Value(1 - y) * (Value(1.0) - p + eps).log())
        loss = loss * (1.0 / len(data))

        for par in model.parameters():
            par.grad = 0.0
        loss.backward()

        for par in model.parameters():
            par.data -= lr * par.grad

        if epoch % 500 == 0:
            print(f"epoch {epoch:4d}  loss={loss.data:.4f}")

    print("\nFinal predictions (scratch engine):")
    for xs, y in data:
        p = model([Value(x) for x in xs])
        r = int(p.data > 0.5)
        ok = "OK" if r == y else "WRONG"
        print(f"  {xs} -> {p.data:.3f} (rounds to {r}, target {y})  {ok}")


if __name__ == "__main__":
    main()
