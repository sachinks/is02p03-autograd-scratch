# IS02P03 — Autograd Scratch

> *"Every neural network is a directed acyclic graph of mathematical operations.
> A forward pass traverses the graph computing outputs from inputs.
> A backward pass traverses in reverse computing gradients via the chain rule.
> `requires_grad=True` tells PyTorch to record each operation for later differentiation."*

**Layer 02 · Representation and Similarity · Chapter 07 of 25**
Knowledge type: **permanent** — backpropagation never changes.

---

## What this project is

A two-layer neural network that learns the **XOR** function, written with **raw
PyTorch tensors only** — no `nn.Module`, no `nn.Linear`, no `F.relu`, no
optimiser. Every line of the forward pass, the backward pass, the weight update
and the gradient reset is written out by hand so the autograd mechanism is
fully visible.

The XOR problem is deliberately trivial. The point is not the problem — it is
the *mechanism*. After this chapter you can look at any PyTorch training loop
and know exactly what every line does at the graph level. That is what lets you
debug gradient bugs, implement LoRA (Chapter 18), and reason about
`requires_grad=False` for frozen layers.

### Files

| File | What it is |
|------|------------|
| `xor_solver.py` | The base build. Raw-tensor 2-layer XOR solver with a manual training loop. |
| `graph_inspect.py` | Prints the `grad_fn` chain so you can *see* the computation graph as a DAG. |
| `micrograd.py` | **Stretch goal.** A ~200-line autograd engine built from scratch in pure Python (no torch), solving XOR with its own `Value` class. This is the Karpathy *micrograd* exercise — proof you understand the mechanism torch gives you for free. |
| `requirements.txt` | `torch`, `numpy`. |

### Run it

```bash
python xor_solver.py      # needs torch
python graph_inspect.py   # needs torch
python micrograd.py       # pure python, no dependencies
```

`xor_solver.py` prints the loss every 200 epochs, then the four final
predictions — all four correct (0,0→0  0,1→1  1,0→1  1,1→0). `micrograd.py`
does the same using the hand-built engine.

---

## What the computation graph is

When you do maths on a tensor that has `requires_grad=True`, PyTorch does two
things at once. First it computes the result normally (the number you want).
Second, it quietly records *how* that result was produced, so it can later run
the calculation backwards and work out gradients.

That record is the **computation graph**. It is a **DAG** — a directed acyclic
graph. Each operation is a **node**. Each node stores three things:

1. the **output tensor** it produced,
2. a reference to the **backward function** for that operation — its `grad_fn`
   (e.g. `MulBackward`, `AddmmBackward`, `SigmoidBackward`),
3. references to its **input tensors** — its parents in the graph.

The edges point *from outputs back to inputs*. That direction is the whole
point: gradients have to flow from the loss at the end, back to the weights at
the start.

A crucial detail: **the graph is not a separate object.** There is no
`model.graph` you can fetch. The graph *lives inside the tensors themselves*.
Every non-leaf tensor carries a `.grad_fn`, and every `grad_fn` carries
`.next_functions` pointing at the `grad_fn`s of its inputs. Following that chain
**is** walking the graph. `graph_inspect.py` does exactly this — it prints the
chain from `output` all the way back to the four leaf parameters.

---

## `requires_grad=True` — physical explanation

This is the BENEATH question, so it deserves the most care.

`requires_grad=True` is a flag on a tensor that says *"track me."* On its own it
does nothing visible. The effect appears the moment the tensor takes part in an
operation. At that point PyTorch:

1. computes the forward result as normal,
2. allocates a C++ **`Function` object** that holds the **backward formula** for
   that specific operation,
3. **saves any inputs** the backward formula will need later (for example, ReLU
   must remember which inputs were positive; sigmoid must remember its own
   output, since its derivative is `s·(1−s)`),
4. stores a pointer to that `Function` object in the result tensor's **`grad_fn`**
   attribute,
5. links that `Function` to its parents through **`next_functions`**.

So `grad_fn` is "the function that *created* this tensor and knows how to
differentiate it," and `next_functions` is "where to send the gradient next."
**The chain of `grad_fn → next_functions` IS the computation graph.** There is
no separate data structure — the graph is woven through the tensors.

**Leaf tensors** are the ones *you* created directly (the weights `W1, b1, W2,
b2`), not ones produced by an operation. Their `grad_fn` is `None`. After
`.backward()`, their gradients land in `.grad`. In the graph these appear as
**`AccumulateGrad`** nodes — the end-points where gradient is deposited.
Non-leaf (intermediate) tensors do *not* keep their gradients by default; they
are just used in transit and then freed.

---

## The three autograd mistakes

These three bugs catch almost everyone. Each one, the error it throws, and the
fix:

### 1. In-place operation on a `requires_grad` tensor

```python
W = torch.randn(4, 4, requires_grad=True)
W += 0.1   # RuntimeError: a leaf Variable that requires grad
           #               has been used in an in-place operation
```

**Why:** in-place editing changes the very tensor that `grad_fn` nodes saved a
reference to. The values saved for the backward pass are now wrong, so PyTorch
refuses. **Fix:** use `W.data += 0.1` (touches storage, not the graph) or
`W = W + 0.1` (makes a new tensor, leaves the old graph intact). In the training
loop we use the `torch.no_grad()` block, which is the clean version of this.

### 2. Forgetting `.grad.zero_()`

```python
for epoch in range(100):
    loss = compute_loss(W)
    loss.backward()        # W.grad += new gradient  (it ADDS, it does not replace)
    W.data -= lr * W.grad
    # W.grad.zero_()  <-- MISSING
```

**Why:** `.backward()` **accumulates** into `.grad` — it adds, never overwrites.
So by epoch 2, `W.grad` holds the *sum* of epoch 1 and epoch 2 gradients, and
your update is garbage. The loss oscillates or diverges. **Fix:** call
`.grad.zero_()` on every parameter before each backward pass (an optimiser's
`zero_grad()` does this for you). The accumulation behaviour is not a bug — it
is what makes gradient accumulation over micro-batches possible — but you must
opt out of it every normal step.

### 3. Premature `.item()` / `.detach()`

```python
loss = compute_loss(W)
loss_val = loss.item()   # returns a plain Python float -> detached from graph
loss_val.backward()      # AttributeError: 'float' object has no attribute 'backward'
```

**Why:** `.item()` converts a tensor to a Python number and throws away
`grad_fn`. `.detach()` makes a tensor with no `grad_fn`, so backward stops dead
there. **Fix:** call `.backward()` *first*, then use `.item()` only for logging
and printing.

(A full failure-modes table — NaN loss, memory leaks, exploding gradients — is
at the bottom of this README.)

---

## The XOR training loop, dissected line by line

```python
y_hat = forward(X)            # FORWARD  - builds a fresh graph this epoch
loss  = bce_loss(y_hat, Y)
loss.backward()               # BACKWARD - walks graph in reverse, fills .grad
with torch.no_grad():         # UPDATE   - outside the graph
    for p in params:
        p -= lr * p.grad
for p in params:
    p.grad.zero_()            # RESET    - clear before next backward
```

**`forward(X)`** — runs `X → Linear → ReLU → Linear → Sigmoid`. Because the
weights have `requires_grad=True`, every operation here adds a node to a brand
new graph. The graph is rebuilt *every epoch* (more on why below).

**`loss.backward()`** — starts at the single loss number and traverses the graph
in **reverse topological order** (outputs before inputs). At each node it
multiplies the incoming gradient by that operation's **local gradient** (the
derivative of the op with respect to its inputs) and passes the result upstream.
This is the chain rule, applied mechanically across the whole graph. The
gradients end up in `W1.grad, b1.grad, W2.grad, b2.grad`. **If you omit this
line, no gradients exist and nothing can learn.**

**`with torch.no_grad():`** — the weight update is *not* part of the model's
computation and must not be differentiated. Without this context, the
subtraction `p -= lr * p.grad` would itself create new graph nodes — a memory
leak and a corrupted graph. `no_grad()` switches off graph recording for
everything inside the block.

**`p -= lr * p.grad`** — gradient descent: nudge each weight a small step `lr`
in the direction that *reduces* the loss (the negative gradient). **Omit it and
the weights never change — no learning.**

**`p.grad.zero_()`** — clears the gradient buffer so the next `.backward()`
starts from zero instead of adding to last epoch's gradient. **Omit it and
gradients accumulate across epochs — the update uses the sum of all past
gradients and training breaks** (mistake #2 above).

**`torch.manual_seed(42)`** (at the top) — fixes the random weight
initialisation so every run is identical. Without it, debugging is impossible
because results change each run.

> **Note on the hidden width:** the chapter uses 4 hidden units; this build uses
> **8**. A 4-neuron ReLU net solves XOR on only ~60% of random seeds — the rest
> get trapped in a *dead-ReLU* local minimum where some neurons output 0 for
> every input, receive zero gradient, and never recover. Eight units give enough
> spare capacity that XOR is solved on every seed. (Verified numerically: 4
> units → 7/12 seeds solved, 8 units → 12/12.)

---

## Vanishing gradients, mechanically

The chain rule **multiplies** local gradients across layers. So the gradient
that reaches an early layer is a *product* of one factor per layer it passed
through.

Sigmoid's derivative is `s·(1−s)`, whose maximum value is **0.25** (at s=0.5).
So every sigmoid layer multiplies the gradient by **at most 0.25**, usually
less. Stack ten sigmoid layers and the gradient reaching layer 1 is at most
`0.25^10 ≈ 0.000001` — effectively zero. The early layers get no useful signal
and stop learning. This is the **vanishing gradient** problem, and it is the
reason sigmoid was replaced by ReLU in deep networks.

**ReLU** mitigates this because its derivative is exactly **1** wherever the
input is positive (and 0 where negative). For the active neurons the gradient
passes through *undiminished* — multiply by 1 as many times as you like and it
does not shrink. ReLU's own weakness is the opposite: a neuron stuck in the
negative region has gradient 0 forever (the *dead neuron* problem — exactly what
forced the move from 4 to 8 hidden units here). Leaky ReLU and GELU patch this
with a small slope on the negative side.

The mirror image is **exploding gradients**: if local gradients are *greater*
than 1 and weights are large, the product grows exponentially and the loss
becomes `NaN`. The standard fix is **gradient clipping**
(`torch.nn.utils.clip_grad_norm_`), applied in essentially every transformer
training run.

---

## LoRA connection (Chapter 18 preview)

LoRA (Low-Rank Adaptation) is the dominant way to fine-tune large language
models, and it is built **entirely** on the mechanism in this chapter.

In a pretrained model you first **freeze** every weight:

```python
for _, param in base_model.named_parameters():
    param.requires_grad = False   # frozen -> no grad_fn -> never updated
```

`requires_grad=False` lifts those tensors out of the computation graph
completely. `.backward()` produces no gradient for them; they are never updated.

LoRA then inserts two small adapter matrices into the forward pass:

```python
# effective weight = W_frozen + (B @ A),  A is (r, d_in),  B is (d_out, r)
A = torch.randn(r, d_in,  requires_grad=True)   # enters the graph
B = torch.zeros(d_out, r, requires_grad=True)   # enters the graph
```

Because only `A` and `B` have `requires_grad=True`, the gradient flows only to
them. With `r` (the rank) much smaller than the dimensions, `A` and `B` together
hold a tiny fraction of the parameters — which is why LoRA trains **0.1–1%** of
the model. The exact same loop you wrote here applies:
forward `x → W_frozen (no grad) + B@A (grad) → output`; backward flows through
`B@A` only; update `A.data -= lr*A.grad; B.data -= lr*B.grad`.

So `requires_grad` is not a detail — it is the switch that decides what gets
trained.

---

## BENEATH — the full answer

**Q: What does `requires_grad=True` physically store as operations chain during
the forward pass? What data structure does PyTorch build, what does
`.backward()` traverse, and why must the graph be recreated every forward pass?**

`requires_grad=True` does not store anything by itself; it is a flag. Its effect
fires on every operation the tensor takes part in. For each such operation
PyTorch allocates a C++ **`Function` object** that holds (a) the **backward
formula** for that operation and (b) any **saved input tensors** that formula
will need (ReLU saves its input sign mask; sigmoid saves its output). A pointer
to this `Function` is written into the result tensor's **`grad_fn`** attribute,
and the `Function` is linked to its parents through **`next_functions`**.

The data structure this produces is a **directed acyclic graph (DAG)**, but it
is **not a separate object** — it is the `grad_fn → next_functions` chain woven
through the tensors themselves. Output tensors point back toward their inputs;
the leaf parameters appear as **`AccumulateGrad`** nodes, the points where
gradients are deposited into `.grad`.

`.backward()` starts at the loss and **traverses this DAG in reverse
topological order** (every node after all the nodes that depend on it). At each
node it multiplies the incoming gradient by the node's **local gradient** and
passes the product upstream — the chain rule applied mechanically. Gradients for
the leaves accumulate into their `.grad`.

The graph must be **recreated on every forward pass** because the nodes hold
references to the intermediate tensors, and after `.backward()` PyTorch **frees
those tensors to reclaim memory**. The next forward pass produces new
intermediate tensors, so it needs a new `grad_fn` chain built around them. This
is PyTorch's **dynamic, define-by-run** graph — the graph is constructed *as the
code runs*, in contrast to a static graph that is compiled once ahead of time
(the classic TensorFlow 1.x model). Dynamic graphs are why ordinary Python
control flow (`if`, `for`) just works inside a model, at the cost of rebuilding
the graph each step.

---

## Failure modes table

| Failure | Symptom | Root cause | Fix |
|---|---|---|---|
| In-place op on `requires_grad` tensor | `RuntimeError` at the in-place line | Saved tensors in `grad_fn` are corrupted | Use `.data` for in-place, or make a new tensor |
| Missing `grad.zero_()` | Loss oscillates / diverges | Gradients from all past epochs accumulate | Call `.zero_()` on every param before each `backward()` |
| Premature `.item()` | `AttributeError: 'float' has no .backward()` | `.item()` detaches from the graph | Call `.backward()` *before* `.item()` |
| Update inside graph (no `no_grad`) | Memory grows every epoch | Update creates new graph nodes | Wrap the update in `torch.no_grad()` |
| NaN loss | Loss is `NaN` immediately | Exploding gradients or `log(0)` in BCE | Gradient clipping + epsilon inside the loss |
| Vanishing gradients | Early layers never learn | Sigmoid gradient ≤ 0.25 per layer | Use ReLU / GELU |
| Wrong `.backward()` count | Gradients are sum of last N losses | Called `backward()` N times without `zero_()` | One `backward()` per step, then `zero_()` |

---

## Status & stretch goals

- [x] **Base (v1.0)** — XOR solver working, three mistakes documented, BENEATH written.
- [x] **Stretch** — from-scratch micrograd engine (`micrograd.py`) solving XOR without torch.
- [ ] Visualise the DAG with `torchviz`/`graphviz`, annotated with op name, shape, `grad_fn`.
- [ ] Gradient-clipping experiment: init weights with `randn * 10`, watch NaN, add `clip_grad_norm_`, document before/after.
- [ ] Implement Adam from scratch (first/second moments, bias correction) and match `torch.optim.Adam`.

**Next chapter:** IS02P04 — Attention Visualiser. The attention weights are
themselves a graph node, so the computation-graph understanding from here
applies directly.

---

MIT © [Sachin Kolige](https://github.com/sachinks)
