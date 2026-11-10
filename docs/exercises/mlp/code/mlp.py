"""
Multi-Layer Perceptron from scratch (NumPy) -- Insper ANN-DL 2026.2, MLP exercise.

Run from the repository root:
    python docs/exercises/mlp/code/mlp.py

It checks the Exercise 1 hand computation, builds the Exercise 2-4 datasets,
trains the MLPs, saves Figures 1-6 to ../figures/ and writes every number the
report quotes to results.json. One generator, rng = np.random.default_rng(42),
is used for every weight initialization and every mini-batch shuffle.
scikit-learn is only used for make_blobs, make_classification,
train_test_split, confusion_matrix and the LogisticRegression baselines.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# The ONLY random generator in the whole script.
rng = np.random.default_rng(42)
R = {}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=130)
    plt.close(fig)


# ===========================================================================
# Exercise 1 -- numerical check of the hand computation (the report shows
# every step by hand; this only confirms the numbers).
# ===========================================================================
x = np.array([0.5, -0.2]); y = 1.0
W1 = np.array([[0.3, -0.1], [0.2, 0.4]]); b1 = np.array([0.1, -0.2])
W2 = np.array([0.5, -0.3]); b2 = 0.2; eta = 0.3

z1 = W1 @ x + b1
h1 = np.tanh(z1)
u2 = W2 @ h1 + b2
yhat = np.tanh(u2)
L = (y - yhat) ** 2
dL_dyhat = -2 * (y - yhat)
dL_du2 = dL_dyhat * (1 - yhat ** 2)
dL_dW2 = dL_du2 * h1
dL_db2 = dL_du2
dL_dh1 = dL_du2 * W2
dL_dz1 = dL_dh1 * (1 - h1 ** 2)
dL_dW1 = np.outer(dL_dz1, x)
dL_db1 = dL_dz1
W1n, b1n = W1 - eta * dL_dW1, b1 - eta * dL_db1
W2n, b2n = W2 - eta * dL_dW2, b2 - eta * dL_db2
L_after = (y - np.tanh(W2n @ np.tanh(W1n @ x + b1n) + b2n)) ** 2
R["ex1"] = {k: np.round(v, 6).tolist() for k, v in {
    "z1": z1, "h1": h1, "u2": u2, "yhat": yhat, "L": L, "dL_dyhat": dL_dyhat,
    "dL_du2": dL_du2, "dL_dW2": dL_dW2, "dL_db2": dL_db2, "dL_dh1": dL_dh1,
    "dL_dz1": dL_dz1, "dL_dW1": dL_dW1, "dL_db1": dL_db1, "W1_new": W1n,
    "b1_new": b1n, "W2_new": W2n, "b2_new": b2n, "L_after": L_after}.items()}


# ===========================================================================
# The MLP (shared by Exercises 2, 3 and 4)
# ===========================================================================
def sigmoid(u):
    return 1 / (1 + np.exp(-u))


def softmax(u):
    e = np.exp(u - u.max(axis=1, keepdims=True))  # shift for numerical stability
    return e / e.sum(axis=1, keepdims=True)


def bce(P, Y):
    """Binary cross-entropy, mean over the batch. P, Y: (N, 1)."""
    P = np.clip(P, 1e-12, 1 - 1e-12)
    return float(-np.mean(Y * np.log(P) + (1 - Y) * np.log(1 - P)))


def cce(P, Y):
    """Categorical cross-entropy, mean over the batch. P, Y: (N, K), Y one-hot."""
    return float(-np.mean(np.sum(Y * np.log(np.clip(P, 1e-12, 1)), axis=1)))


class MLP:
    """Fully connected network: tanh hidden layers, sigmoid or softmax output.

    sizes  -- e.g. [2, 8, 1] or [4, 32, 32, 3] (input, hidden..., output)
    output -- "sigmoid" (pairs with BCE) or "softmax" (pairs with CCE)
    For both pairs the gradient of the loss w.r.t. the output pre-activation
    is (P - Y) / N, so backward() is identical for Exercises 2, 3 and 4.
    """

    def __init__(self, sizes, output, rng):
        self.sizes, self.out = sizes, {"sigmoid": sigmoid, "softmax": softmax}[output]
        # Xavier/Glorot-normal init (suited to tanh), zero biases
        self.W = [rng.normal(0, np.sqrt(2 / (a + b)), size=(a, b))
                  for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]

    def n_params(self):
        return int(sum(W.size + b.size for W, b in zip(self.W, self.b)))

    def forward(self, X):
        """Returns output probabilities; stores every layer's activation."""
        self.A = [X]
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            U = self.A[-1] @ W + b
            last = l == len(self.W) - 1
            self.A.append(self.out(U) if last else np.tanh(U))
        return self.A[-1]

    def backward(self, Y):
        """Gradients of the mean loss, by the chain rule, layer by layer."""
        N = len(Y)
        delta = (self.A[-1] - Y) / N              # dL/dU at the output layer
        self.gW, self.gb = [None] * len(self.W), [None] * len(self.W)
        for l in range(len(self.W) - 1, -1, -1):
            self.gW[l] = self.A[l].T @ delta       # dL/dW = a_prev^T delta
            self.gb[l] = delta.sum(axis=0)         # dL/db = sum of delta
            if l > 0:                              # through W, then tanh'
                delta = (delta @ self.W[l].T) * (1 - self.A[l] ** 2)

    def step(self, lr):
        for l in range(len(self.W)):
            self.W[l] -= lr * self.gW[l]
            self.b[l] -= lr * self.gb[l]


def train(model, X, Y, loss_fn, lr, epochs, batch, rng):
    """Mini-batch gradient descent; returns the full training loss per epoch."""
    history = []
    for _ in range(epochs):
        order = rng.permutation(len(X))
        for i in range(0, len(X), batch):
            idx = order[i:i + batch]
            model.forward(X[idx])
            model.backward(Y[idx])
            model.step(lr)
        history.append(loss_fn(model.forward(X), Y))
    return np.array(history)


def one_hot(y, k):
    return np.eye(k)[y]


# ===========================================================================
# Exercise 2 -- binary, one class in the middle, the other on both sides
# ===========================================================================
X2, cl = make_blobs(n_samples=[500, 250, 250], centers=[[0, 0], [3, 3], [-3, -3]],
                    cluster_std=1.2, random_state=42)
y2 = (cl != 0).astype(int)                       # clusters 2 and 3 -> class 1
X2tr, X2te, y2tr, y2te = train_test_split(X2, y2, test_size=0.2, random_state=42, stratify=y2)

fig, ax = plt.subplots(figsize=(6.5, 5.5))
for c, lab, col in [(0, "Class 0 (center cluster)", "tab:blue"),
                    (1, "Class 1 (two outer clusters)", "tab:orange")]:
    ax.scatter(*X2[y2 == c].T, s=10, alpha=0.6, c=col, label=lab)
ax.set(title="Figure 1. Exercise 2 data (1000 points)", xlabel="x1", ylabel="x2")
ax.legend()
save(fig, "fig1.png")

base2 = LogisticRegression().fit(X2tr, y2tr)
R["ex2_baseline_acc"] = float(base2.score(X2te, y2te))
R["ex2_baseline_coef"] = base2.coef_.ravel().tolist() + [float(base2.intercept_[0])]

EX2 = dict(sizes=[2, 8, 1], lr=0.1, epochs=500, batch=32)
mlp2 = MLP(EX2["sizes"], "sigmoid", rng)
hist2 = train(mlp2, X2tr, y2tr[:, None].astype(float), bce, EX2["lr"], EX2["epochs"], EX2["batch"], rng)
pred2 = (mlp2.forward(X2te)[:, 0] >= 0.5).astype(int)
R["ex2_mlp"] = {**EX2, "params": mlp2.n_params(), "test_acc": float((pred2 == y2te).mean()),
                "train_acc": float(((mlp2.forward(X2tr)[:, 0] >= 0.5) == y2tr).mean()),
                "loss_first": float(hist2[0]), "loss_last": float(hist2[-1])}

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(np.arange(1, len(hist2) + 1), hist2, c="black", label="training BCE")
ax.set(title="Figure 2. Exercise 2: training loss vs. epoch", xlabel="epoch",
       ylabel="binary cross-entropy (train)")
ax.legend()
save(fig, "fig2.png")

gx, gy = np.meshgrid(np.linspace(-8, 8, 400), np.linspace(-8, 8, 400))
prob = mlp2.forward(np.c_[gx.ravel(), gy.ravel()])[:, 0].reshape(gx.shape)
fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.contourf(gx, gy, (prob >= 0.5).astype(int), levels=[-0.5, 0.5, 1.5],
            colors=["tab:blue", "tab:orange"], alpha=0.2)
ax.contour(gx, gy, prob, levels=[0.5], colors="black", linewidths=1.2)
for c, lab, col in [(0, "Class 0", "tab:blue"), (1, "Class 1", "tab:orange")]:
    ax.scatter(*X2te[y2te == c].T, s=12, c=col, edgecolors="white", linewidths=0.3,
               label=f"{lab} (test)")
# the linear baseline's boundary, for comparison
w1, w2, w0 = R["ex2_baseline_coef"]
xs = np.linspace(-8, 8, 2)
ax.plot(xs, -(w0 + w1 * xs) / w2, "--", c="gray", label="logistic regression boundary")
ax.set(title="Figure 3. Exercise 2: MLP decision regions", xlabel="x1", ylabel="x2",
       xlim=(-8, 8), ylim=(-8, 8))
ax.legend(loc="upper left", fontsize=8)
save(fig, "fig3.png")


# ===========================================================================
# Exercise 3 -- three classes, 4 features, 2 clusters per class
# ===========================================================================
X3, y3 = make_classification(n_samples=1500, n_features=4, n_informative=4, n_redundant=0,
                             n_repeated=0, n_classes=3, n_clusters_per_class=2,
                             class_sep=1.0, random_state=42)
X3tr, X3te, y3tr, y3te = train_test_split(X3, y3, test_size=0.2, random_state=42, stratify=y3)
Y3tr = one_hot(y3tr, 3)

base3 = LogisticRegression(max_iter=1000).fit(X3tr, y3tr)
R["ex3_baseline_acc"] = float(base3.score(X3te, y3te))

HP = dict(lr=0.05, epochs=300, batch=32)   # shared by Exercises 3 and 4
SIZES3 = [4, 32, 3]
SIZES4 = [4, 32, 32, 3]


def run(sizes):
    m = MLP(sizes, "softmax", rng)
    h = train(m, X3tr, Y3tr, cce, HP["lr"], HP["epochs"], HP["batch"], rng)
    pred = m.forward(X3te).argmax(1)
    return m, h, pred


mlp3, hist3, pred3 = run(SIZES3)
cm = confusion_matrix(y3te, pred3)
off = cm + cm.T
np.fill_diagonal(off, 0)
i, j = np.unravel_index(off.argmax(), off.shape)
R["ex3_mlp"] = {"sizes": SIZES3, **HP, "params": mlp3.n_params(),
                "test_acc": float((pred3 == y3te).mean()), "final_train_loss": float(hist3[-1]),
                "confusion": cm.tolist(), "most_confused": [int(min(i, j)), int(max(i, j))],
                "most_confused_count": int(off[i, j])}
R["ex3_baseline_confusion"] = confusion_matrix(y3te, base3.predict(X3te)).tolist()

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(np.arange(1, len(hist3) + 1), hist3, c="black", label="training CCE")
ax.set(title="Figure 4. Exercise 3: training loss vs. epoch", xlabel="epoch",
       ylabel="categorical cross-entropy (train)")
ax.legend()
save(fig, "fig4.png")

fig, ax = plt.subplots(figsize=(5.5, 4.8))
im = ax.imshow(cm, cmap="Blues")
for a in range(3):
    for b in range(3):
        ax.text(b, a, cm[a, b], ha="center", va="center",
                color="white" if cm[a, b] > cm.max() / 2 else "black")
ax.set(title="Figure 5. Exercise 3: test confusion matrix", xlabel="predicted class",
       ylabel="true class", xticks=range(3), yticks=range(3))
fig.colorbar(im, ax=ax, label="test points")
save(fig, "fig5.png")


# ===========================================================================
# Exercise 4 -- same data and budget, only depth changes
# ===========================================================================
mlp4, hist4, pred4 = run(SIZES4)
R["ex4_mlp"] = {"sizes": SIZES4, **HP, "params": mlp4.n_params(),
                "test_acc": float((pred4 == y3te).mean()),
                "final_train_loss": float(hist4[-1])}

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(np.arange(1, HP["epochs"] + 1), hist3, label=f"Exercise 3: {SIZES3} (1 hidden)")
ax.plot(np.arange(1, HP["epochs"] + 1), hist4, label=f"Exercise 4: {SIZES4} (2 hidden)")
ax.set(title="Figure 6. Training loss, 1 vs. 2 hidden layers", xlabel="epoch",
       ylabel="categorical cross-entropy (train)")
ax.legend()
save(fig, "fig6.png")

# Several seeds: each run takes its initialization and shuffles from the same
# rng, so runs 1..5 differ only in where the rng stream is (5 >= 3 seeds).
N_RUNS = 5
multi = {"1 hidden": [], "2 hidden": []}
for r in range(N_RUNS):
    for name, sizes in [("1 hidden", SIZES3), ("2 hidden", SIZES4)]:
        m, h, p = run(sizes)
        multi[name].append({"test_acc": float((p == y3te).mean()), "final_train_loss": float(h[-1]),
                            "train_acc": float((m.forward(X3tr).argmax(1) == y3tr).mean())})
R["ex4_multi"] = {
    name: {"runs": v,
           "mean_test_acc": float(np.mean([d["test_acc"] for d in v])),
           "std_test_acc": float(np.std([d["test_acc"] for d in v])),
           "mean_final_train_loss": float(np.mean([d["final_train_loss"] for d in v])),
           "mean_train_acc": float(np.mean([d["train_acc"] for d in v]))}
    for name, v in multi.items()
}
R["ex4_mean_diff_acc"] = R["ex4_multi"]["2 hidden"]["mean_test_acc"] - R["ex4_multi"]["1 hidden"]["mean_test_acc"]

# ---------------------------------------------------------------------------
# Gradient check (done last so it does not change the rng stream above):
# compare backward() with central finite differences on a small batch.
def grad_check(sizes, output, loss_fn, k):
    m = MLP(sizes, output, rng)
    Xc = rng.normal(size=(5, sizes[0]))
    Yc = one_hot(rng.integers(0, k, 5), k) if output == "softmax" else rng.integers(0, 2, (5, 1)).astype(float)
    m.forward(Xc); m.backward(Yc)
    worst, eps = 0.0, 1e-6
    for P, G in list(zip(m.W, m.gW)) + list(zip(m.b, m.gb)):
        for idx in np.ndindex(P.shape):
            old = P[idx]
            P[idx] = old + eps; lp = loss_fn(m.forward(Xc), Yc)
            P[idx] = old - eps; lm = loss_fn(m.forward(Xc), Yc)
            P[idx] = old
            num = (lp - lm) / (2 * eps)
            worst = max(worst, abs(num - G[idx]) / max(1e-8, abs(num) + abs(G[idx])))
    return worst

R["grad_check_max_rel_error"] = {"sigmoid+BCE [2,8,1]": grad_check([2, 8, 1], "sigmoid", bce, 2),
                                 "softmax+CCE [4,32,32,3]": grad_check([4, 32, 32, 3], "softmax", cce, 3)}

with open(HERE / "results.json", "w") as f:
    json.dump(R, f, indent=2)
print(json.dumps(R, indent=2))
