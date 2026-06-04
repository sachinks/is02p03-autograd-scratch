"""
micrograd.py - a tiny autograd engine built FROM SCRATCH (no torch).

This is the Karpathy "micrograd" exercise. It re-implements, in ~90 lines of
plain Python, exactly what PyTorch's autograd does:

  * Value  = one node in the computation graph (a scalar + its gradient).
  * Each math operation (+, *, **, relu, sigmoid, ...) returns a NEW Value
    and records (a) its parents and (b) a local _backward closure that knows
    how to push gradient onto those parents via the chain rule.
  * .backward() does a topological sort of the graph, then applies every
    _backward closure in reverse order - this is what PyTorch's C++ engine
    does, just slower and in Python.

Solving XOR with this engine proves you understand the mechanism that
xor_solver.py gets "for free" from torch.

Run:  python micrograd.py
"""

import math
import random


class Value:
    """A single scalar value and its gradient, plus its place in the graph."""

    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None     # how to send grad to parents
        self._prev = set(_children)       # parent Values (graph edges)
        self._op = _op                    # label, for debugging only

    # ---- operations: each builds a new node + its local backward rule ----
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(out)/d(self) = 1, d(out)/d(other) = 1  -> just pass grad through
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # product rule: d(a*b)/da = b, d(a*b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, p):
        assert isinstance(p, (int, float))
        out = Value(self.data ** p, (self,), f"**{p}")

        def _backward():
            self.grad += (p * self.data ** (p - 1)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():
            # gradient flows only where the input was positive
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, (self,), "sigmoid")

        def _backward():
            # d(sigmoid)/dx = sigmoid * (1 - sigmoid)
            self.grad += (s * (1 - s)) * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    # ---- the engine: topological order, then chain rule in reverse ----
    def backward(self):
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for parent in v._prev:
                    build(parent)
                topo.append(v)
        build(self)

        self.grad = 1.0                 # d(self)/d(self) = 1, the seed
        for node in reversed(topo):     # outputs before inputs
            node._backward()

    # ---- right-hand-side and negation helpers ----
    def __neg__(self):        return self * -1
    def __radd__(self, o):    return self + o
    def __sub__(self, o):     return self + (-o)
    def __rsub__(self, o):    return (-self) + o
    def __rmul__(self, o):    return self * o
    def __truediv__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        return self * o ** -1
    def __repr__(self):       return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


# ----------------------------------------------------------------------
# A 2-4-1 MLP built from Value objects, trained on XOR.
# ----------------------------------------------------------------------
class Neuron:
    def __init__(self, n_in, activation):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0.0)
        self.activation = activation

    def __call__(self, xs):
        act = sum((wi * xi for wi, xi in zip(self.w, xs)), self.b)
        return act.relu() if self.activation == "relu" else act.sigmoid()

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, n_in, n_out, activation):
        self.neurons = [Neuron(n_in, activation) for _ in range(n_out)]

    def __call__(self, xs):
        return [n(xs) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self):
        # 8 hidden units (not 4): a narrow ReLU net often gets stuck because
        # some neurons "die" (output 0 for every input -> zero gradient ->
        # never recover). Extra width gives the network spare capacity so at
        # least a few neurons stay alive and XOR is solved on every seed.
        self.l1 = Layer(2, 8, "relu")
        self.l2 = Layer(8, 1, "sigmoid")

    def __call__(self, xs):
        return self.l2(self.l1(xs))[0]   # single output neuron

    def parameters(self):
        return self.l1.parameters() + self.l2.parameters()


def main():
    random.seed(42)
    data = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]
    model = MLP()
    lr = 0.5

    for epoch in range(3000):
        # forward + binary cross-entropy loss over the 4 examples
        loss = Value(0.0)
        eps = 1e-7
        for xs, y in data:
            p = model([Value(x) for x in xs])
            loss = loss + -(Value(y) * (p + eps).log()
                            + Value(1 - y) * (Value(1.0) - p + eps).log())
        loss = loss * (1.0 / len(data))

        # zero grads (our Values accumulate too), then backward
        for par in model.parameters():
            par.grad = 0.0
        loss.backward()

        # manual SGD update
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
