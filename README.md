# IS02P03 — Autograd Scratch

> *"Every neural network is a directed acyclic graph of mathematical operations. A forward pass traverses the graph computing outputs from inputs. A backward pass traverses in reverse computing gradients via the chain rule. requires_grad=True tells PyTorch to record each operation for later differentiation."*

---

## What this project builds

A two-layer neural network that learns the XOR function, written with **raw PyTorch tensors only** — no `nn.Module`, no `nn.Linear`, no `F.relu`, no optimiser. Every line of the forward pass, the backward pass, the weight update, and the gradient reset is written out by hand so the autograd mechanism is fully visible.

The XOR problem is deliberately trivial. The point is not the problem — it is the *mechanism*. After this chapter you can look at any PyTorch training loop and know exactly what every line does at the graph level. That understanding is what lets you debug gradient bugs, implement LoRA (Chapter 18), and reason about `requires_grad=False` for frozen layers.

A stretch goal goes further: `micrograd.py` is a ~200-line autograd engine built from scratch in pure Python with no dependencies, solving XOR using its own `Value` class. This is the Karpathy micrograd exercise — proof that the mechanism PyTorch gives you for free is not magic.

---

## The computation graph

When you do maths on a tensor that has `requires_grad=True`, PyTorch does two things at once. First it computes the result normally. Second, it quietly records *how* that result was produced so it can later run the calculation backwards and compute gradients.

That record is the **computation graph** — a **directed acyclic graph (DAG)**. Each operation is a node. Each node stores three things:

1. the **output tensor** it produced,
2. a reference to the **backward function** for that operation (`grad_fn` — e.g. `MulBackward`, `SigmoidBackward`),
3. references to its **input tensors** — its parents in the graph.

The edges point *from outputs back to inputs*. That direction is the whole point: gradients have to flow from the loss at the end, back to the weights at the start.

A crucial detail: **the graph is not a separate object.** There is no `model.graph` you can fetch. The graph *lives inside the tensors themselves* — every non-leaf tensor carries a `.grad_fn`, and every `grad_fn` carries `.next_functions` pointing at the `grad_fn`s of its inputs. Following that chain is walking the graph. `graph_inspect.py` does exactly this.

---

## `requires_grad=True` — physical explanation

`requires_grad=True` is a flag on a tensor that says *"track me."* Its effect fires on every operation the tensor takes part in. For each such operation PyTorch:

1. computes the forward result as normal,
2. allocates a C++ **`Function` object** holding the **backward formula** for that specific operation,
3. **saves any inputs** the backward formula will need later (ReLU must remember which inputs were positive; sigmoid must remember its own output, since its derivative is `s·(1−s)`),
4. stores a pointer to that `Function` in the result tensor's **`grad_fn`**,
5. links that `Function` to its parents through **`next_functions`**.

**Leaf tensors** are the ones you created directly (the weights `W1, b1, W2, b2`), not ones produced by an operation. Their `grad_fn` is `None`. After `.backward()`, their gradients land in `.grad`. In the graph they appear as `AccumulateGrad` nodes — the end-points where gradient is deposited. Non-leaf tensors do not keep their gradients by default; they are used in transit and freed.

---

## The three autograd mistakes

### 1. In-place operation on a `requires_grad` tensor

```python
W = torch.randn(4, 4, requires_grad=True)
W += 0.1   # RuntimeError: a leaf Variable that requires grad
           #               has been used in an in-place operation
```

In-place editing changes the very tensor that `grad_fn` nodes saved a reference to. **Fix:** use `W.data += 0.1` (touches storage, not the graph) or `W = W + 0.1` (makes a new tensor). In the training loop we use `torch.no_grad()`, which is the clean version of this.

### 2. Forgetting `grad.zero_()`

```python
for epoch in range(100):
    loss = compute_loss(W)
    loss.backward()        # W.grad += new gradient — it ADDS, does not replace
    W.data -= lr * W.grad
    # W.grad.zero_()  <-- MISSING
```

`.backward()` **accumulates** into `.grad` — it adds, never overwrites. By epoch 2, `W.grad` holds the sum of epoch 1 and epoch 2 gradients, and the update is garbage. **Fix:** call `.zero_()` on every parameter before each backward pass.

### 3. Premature `.item()` or `.detach()`

```python
loss_val = loss.item()   # returns a plain Python float — detached from graph
loss_val.backward()      # AttributeError: 'float' has no .backward()
```

`.item()` converts a tensor to a Python number and discards `grad_fn`. **Fix:** call `.backward()` first, then use `.item()` only for logging.

---

## Vanishing gradients, mechanically

The chain rule **multiplies** local gradients across layers. Sigmoid's derivative is `s·(1−s)`, whose maximum is **0.25** (at s=0.5). Stack ten sigmoid layers and the gradient reaching layer 1 is at most `0.25^10 ≈ 0.000001` — effectively zero. Early layers get no signal and stop learning.

**ReLU** mitigates this because its derivative is exactly **1** wherever the input is positive. For active neurons, the gradient passes through undiminished. ReLU's own weakness is the opposite: a neuron stuck in the negative region has gradient 0 forever — the *dead neuron* problem, which forced the move from 4 to 8 hidden units in this project (see Observed).

The mirror image is **exploding gradients**: if local gradients are greater than 1 and weights are large, the product grows exponentially and the loss becomes `NaN`. The standard fix is **gradient clipping** (`torch.nn.utils.clip_grad_norm_`), applied in essentially every transformer training run.

---

## LoRA connection (Chapter 18 preview)

LoRA (Low-Rank Adaptation) is built entirely on the mechanism in this chapter. In a pretrained model, first freeze every weight:

```python
for _, param in base_model.named_parameters():
    param.requires_grad = False   # lifted out of the graph — never updated
```

LoRA then inserts two small adapter matrices:

```python
A = torch.randn(r, d_in,  requires_grad=True)   # enters the graph
B = torch.zeros(d_out, r, requires_grad=True)   # enters the graph
```

Because only `A` and `B` have `requires_grad=True`, gradients flow only to them. With rank `r` much smaller than the model dimensions, `A` and `B` together hold a tiny fraction of the parameters — which is why LoRA trains 0.1–1% of the model. The exact same loop you wrote here applies. `requires_grad` is not a detail — it is the switch that decides what gets trained.

---

## How to install & run

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the XOR solver (needs torch)
python xor_solver.py
# prints loss every 200 epochs, then 4 final predictions — all correct

# 4. Print the computation graph DAG (needs torch)
python graph_inspect.py

# 5. Run the from-scratch autograd engine (pure Python, no dependencies)
python micrograd.py

# 6. Render the computation graph as a PNG image (stretch goal 1)
pip install torchviz
sudo apt install graphviz   # system-level, needed by torchviz
python visualise_graph.py
# saves xor_graph.png in the project directory
```

`torch.manual_seed(42)` is set at the top of all three torch files — every run is identical.

---

## Project structure

```
is02p03-autograd-scratch/
  xor_solver.py        raw-tensor 2-layer XOR solver, manual training loop
  graph_inspect.py     prints the grad_fn DAG via next_functions recursion
  micrograd.py         from-scratch autograd engine (~200 lines, pure Python)
  visualise_graph.py   renders the computation graph as a PNG via torchviz (stretch goal 1)
  requirements.txt     torch, numpy
  README.md
```

---

## Algorithm & code flow

### `xor_solver.py` — the training loop

**Dataset and parameters.** `X` (shape `(4, 2)`) and `Y` (shape `(4, 1)`) are plain float tensors — no `requires_grad`. Weights are initialised with `torch.manual_seed(42)` then `torch.randn` / `torch.zeros`:

```
W1 = torch.randn(2, HIDDEN, requires_grad=True)   # (2, 8)
b1 = torch.zeros(HIDDEN,    requires_grad=True)   # (8,)  — zeros init
W2 = torch.randn(HIDDEN, 1, requires_grad=True)   # (8, 1)
b2 = torch.zeros(1,         requires_grad=True)   # (1,)  — zeros init
```

`params = [W1, b1, W2, b2]`, `lr = 0.1`, `EPOCHS = 2000`.

**`relu(x)`** — implemented as `torch.clamp(x, min=0.0)`, not `torch.relu()`. Both are equivalent but `clamp` makes the "floor at zero" literal.

**`sigmoid(x)`** — implemented as `1.0 / (1.0 + torch.exp(-x))`, not `torch.sigmoid()`. Same reason — makes the formula visible.

**`forward(x)`** — four tensor ops, each adding a node to the graph:
```
z1 = x @ W1 + b1      # (4,2)@(2,8) + (8,) → (4,8)  bias broadcasts
a1 = relu(z1)          # (4,8) element-wise
z2 = a1 @ W2 + b2      # (4,8)@(8,1) + (1,) → (4,1)
return sigmoid(z2)     # (4,1) probabilities
```

**`bce_loss(y_hat, y_true)`** — `eps = 1e-7` is added inside both `log()` calls to prevent `log(0) = -inf` → NaN loss when a prediction saturates to 0 or 1.

**`train()`** — four-step loop for `EPOCHS = 2000`:
1. `y_hat = forward(X)` + `loss = bce_loss(y_hat, Y)` — builds fresh graph
2. `loss.backward()` — fills `.grad` on all four params
3. `with torch.no_grad(): p -= lr * p.grad` — update outside graph
4. `p.grad.zero_()` for each param — clear before next step

Prints `f"epoch {epoch:4d}  loss={loss.item():.4f}"` every 200 epochs.

**`evaluate()`** — runs `forward(X)` inside `torch.no_grad()`, rounds each output to 0/1 with `int(p.item() > 0.5)`, prints OK/WRONG per input.

### `graph_inspect.py` — DAG printer

Architecture matches `xor_solver.py`: `HIDDEN = 8`, `torch.manual_seed(42)`. The forward pass uses `torch.relu()` and `torch.sigmoid()` (library calls, not custom functions) — the grad_fn names in the output reflect this.

The file first prints the `.grad_fn` of each intermediate tensor:
```
x.grad_fn      : None        ← leaf
W1.grad_fn     : None        ← leaf parameter (2x8)
z1.grad_fn     : <AddmmBackward0 ...>
a1.grad_fn     : <ReluBackward0 ...>
z2.grad_fn     : <AddmmBackward0 ...>
output.grad_fn : <SigmoidBackward0 ...>
```

Then calls `walk(output.grad_fn)` — a recursive function that prints `type(fn).__name__` indented by `"    " * depth`, then iterates `getattr(fn, "next_functions", ())` (safe getattr for leaf nodes that have no `next_functions`) and recurses on each `parent` from the `(parent, output_index)` pairs. The tree terminates at `AccumulateGrad` nodes (the four leaf parameters). A final line prints actual weight shapes for confirmation.

### `micrograd.py` — from-scratch autograd engine

**`Value` class** wraps a single Python float with `data`, `grad = 0.0`, `_backward = lambda: None`, `_prev = set(_children)`, and `_op` (debug label). Every operator builds a new `Value` node and sets its `_backward` closure:

| Op | Forward | `_backward` formula |
|---|---|---|
| `__add__` | `self.data + other.data` | `self.grad += out.grad; other.grad += out.grad` |
| `__mul__` | `self.data * other.data` | `self.grad += other.data * out.grad; other.grad += self.data * out.grad` |
| `__pow__` | `self.data ** p` | `self.grad += p * self.data**(p-1) * out.grad` |
| `relu` | `max(0, self.data)` | `self.grad += (self.data > 0) * out.grad` |
| `sigmoid` | `1/(1+exp(-x))` | `self.grad += s*(1-s) * out.grad` |
| `log` | `math.log(self.data)` | `self.grad += (1/self.data) * out.grad` |

Helper operators (`__neg__`, `__radd__`, `__sub__`, `__rsub__`, `__rmul__`, `__truediv__`) are all derived from the core six — e.g. `__truediv__` is `self * other**-1`.

**`backward()`** — builds topological order via a DFS `build(v)` helper (post-order: `topo.append(v)` after visiting all parents in `v._prev`). Then sets `self.grad = 1.0` and iterates `reversed(topo)` calling `node._backward()` on each.

**`Neuron(n_in, activation)`** — weights `[Value(random.uniform(-1,1)) for _ in range(n_in)]`, bias `Value(0.0)`. `__call__` computes `sum((wi*xi for wi,xi in zip(w, xs)), self.b)` — the `self.b` as the `start` argument to `sum()` avoids adding a zero-init `Value(0)`. Applies `relu()` or `sigmoid()` based on `activation`.

**`MLP`** — `l1 = Layer(2, 8, "relu")`, `l2 = Layer(8, 1, "sigmoid")`. `__call__` returns `self.l2(self.l1(xs))[0]` — `[0]` extracts the single output `Value` from the length-1 list returned by `Layer`.

**`main()`** — `random.seed(42)`, `lr = 0.5`, 3000 epochs (vs torch: `manual_seed(42)`, `lr = 0.1`, 2000). Slower because arithmetic is scalar-by-scalar. Loop: accumulate BCE loss over 4 examples into `loss = Value(0.0)`, divide by `len(data)`, zero all `.grad` fields manually, `loss.backward()`, SGD `par.data -= lr * par.grad`. Prints every 500 epochs.

### `visualise_graph.py` — stretch goal 1: computation graph PNG

Renders the same DAG that `graph_inspect.py` prints as text into a visual PNG image using `torchviz.make_dot` and Graphviz.

**Setup.** Same architecture as `xor_solver.py` — `HIDDEN = 8`, `torch.manual_seed(42)`, same shapes. A single forward pass (no training) builds the computation graph.

**`make_dot(loss, params=params)`** — walks the `grad_fn → next_functions` chain from the loss tensor. The `params` dict maps display names (with shapes) to leaf tensors so they appear labelled in the diagram instead of showing raw memory addresses:

```python
params = {
    "W1 (2, 8)": W1,
    "b1 (8,)":   b1,
    "W2 (8, 1)": W2,
    "b2 (1,)":   b2,
}
```

**Output (`xor_graph.png`)** — a top-to-bottom DAG (`rankdir="TB"`) showing:
- **Light-blue nodes** — leaf parameters (`W1`, `b1`, `W2`, `b2`) with their shapes
- **Grey nodes** — operation nodes labelled with `grad_fn` type (`MmBackward0`, `ClampBackward1`, `ExpBackward0`, etc.)
- **Green node** — the scalar loss output at the bottom
- **Arrows** — gradient flow direction, from loss back to weights

The chain reads: `Mm → Add → Clamp(ReLU) → Mm → Add → Neg → Exp → Add → Reciprocal(sigmoid) → Mul/Log(BCE) → Mean → Neg → loss`.

`show_attrs=False, show_saved=False` keep the diagram uncluttered. `cleanup=True` removes the intermediate `.gv` dot source file.

---

## Observed

**XOR convergence.** `xor_solver.py` converges cleanly with `torch.manual_seed(42)` and HIDDEN=8. Loss reaches 0.0034 after 2000 epochs. All four predictions are correct: 0,0→0 · 0,1→1 · 1,0→1 · 1,1→0.

**Dead-ReLU with 4 hidden units.** The chapter uses HIDDEN=4. A 4-neuron ReLU net solves XOR on only ~60% of random seeds. The rest get trapped: some neurons receive negative pre-activation for every input, output 0, receive zero gradient, and never recover — dead neurons. Verified numerically: 4 units → 7/12 seeds solved, 8 units → 12/12. Both `xor_solver.py` and `micrograd.py` use HIDDEN=8 for this reason.

**micrograd parity.** `micrograd.py` converges to the same result as `xor_solver.py` using only Python arithmetic — no torch, no numpy. The loss curve shape and final predictions match. This confirms the mechanism is identical: the difference between micrograd and PyTorch is C++ speed and GPU support, not the algorithm.

**Failure modes reference**

| Failure | Symptom | Root cause | Fix |
|---|---|---|---|
| In-place op on `requires_grad` tensor | `RuntimeError` at the in-place line | Saved tensors in `grad_fn` corrupted | Use `.data` for in-place, or make a new tensor |
| Missing `grad.zero_()` | Loss oscillates / diverges | Gradients accumulate across epochs | Call `.zero_()` on every param before each `backward()` |
| Premature `.item()` | `AttributeError: float has no .backward()` | `.item()` detaches from graph | Call `.backward()` before `.item()` |
| Update inside graph (no `no_grad`) | Memory grows every epoch | Update creates new graph nodes | Wrap update in `torch.no_grad()` |
| NaN loss | Loss is `NaN` immediately | Exploding gradients or `log(0)` in BCE | Gradient clipping + epsilon in loss |
| Vanishing gradients | Early layers never learn | Sigmoid gradient ≤ 0.25 per layer | Use ReLU / GELU |

---

## BENEATH

**What does `requires_grad=True` physically store as operations chain during the forward pass? What data structure does PyTorch build, what does `.backward()` traverse, and why must the graph be recreated every forward pass?**

`requires_grad=True` does not store anything by itself; it is a flag. Its effect fires on every operation the tensor takes part in. For each such operation PyTorch allocates a C++ **`Function` object** that holds (a) the **backward formula** for that operation and (b) any **saved input tensors** that formula will need (ReLU saves its input sign mask; sigmoid saves its output). A pointer to this `Function` is written into the result tensor's **`grad_fn`** attribute, and the `Function` is linked to its parents through **`next_functions`**.

The data structure this produces is a **directed acyclic graph (DAG)**, but it is **not a separate object** — it is the `grad_fn → next_functions` chain woven through the tensors themselves. Output tensors point back toward their inputs; the leaf parameters appear as **`AccumulateGrad`** nodes, the points where gradients are deposited into `.grad`.

`.backward()` starts at the loss and **traverses this DAG in reverse topological order** (every node after all nodes that depend on it). At each node it multiplies the incoming gradient by the node's **local gradient** and passes the product upstream — the chain rule applied mechanically. Gradients for the leaves accumulate into their `.grad`.

The graph must be **recreated on every forward pass** because the nodes hold references to intermediate tensors, and after `.backward()` PyTorch **frees those tensors to reclaim memory**. The next forward pass produces new intermediate tensors, so it needs a new `grad_fn` chain built around them. This is PyTorch's **dynamic, define-by-run** graph — constructed as the code runs, in contrast to a static graph compiled once ahead of time (TensorFlow 1.x). Dynamic graphs are why ordinary Python control flow (`if`, `for`) just works inside a model, at the cost of rebuilding the graph each step.

---

## License

MIT © [Sachin Kolige](https://github.com/sachinks)
