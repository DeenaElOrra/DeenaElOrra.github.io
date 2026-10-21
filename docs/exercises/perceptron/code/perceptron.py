"""
Perceptron from scratch (NumPy only) -- Insper ANN-DL 2026.2, Exercise 2.

Run from the repository root:
    python docs/exercises/perceptron/code/perceptron.py

It generates both datasets, trains the perceptron, prints every number the
report quotes, saves Figures 1-6 to ../figures/ and writes results.json.
A single generator, rng = np.random.default_rng(42), is used for everything,
so every run produces exactly the same numbers and figures.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # no display needed, we only save PNGs
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# The ONLY random generator in the whole script.
rng = np.random.default_rng(42)

ETA = 0.01          # learning rate from the statement
MAX_EPOCHS = 100    # epoch cap from the statement
N_PER_CLASS = 1000  # samples per class


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def make_dataset(mean0, cov0, mean1, cov1, n=N_PER_CLASS):
    """Draw n points per class from two 2D Gaussians and shuffle them.

    Class 0 gets label 0 and class 1 gets label 1. The rows are shuffled once
    (with the same rng) so the training loop does not see all class-0 points
    first and then all class-1 points.
    """
    x0 = rng.multivariate_normal(mean0, cov0, size=n)
    x1 = rng.multivariate_normal(mean1, cov1, size=n)
    X = np.vstack([x0, x1])
    y = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    order = rng.permutation(len(y))
    return X[order], y[order]


# ---------------------------------------------------------------------------
# The perceptron (activation, prediction, update rule, training loop)
# ---------------------------------------------------------------------------
def step(z):
    """Heaviside activation: 1 if z >= 0, else 0."""
    return (z >= 0).astype(int)


def predict(X, w, b):
    """y_hat = step(w . x + b) for every row of X."""
    return step(X @ w + b)


def accuracy(X, y, w, b):
    """Fraction of points of the full dataset classified correctly."""
    return float(np.mean(predict(X, w, b) == y))


def train_perceptron(X, y, w0, b0=0.0, eta=ETA, max_epochs=MAX_EPOCHS):
    """Online perceptron with the {0,1} error-driven rule and a pocket.

    For each sample (x, y):
        y_hat = step(w . x + b)
        w <- w + eta * (y - y_hat) * x
        b <- b + eta * (y - y_hat)
    (y - y_hat) is 0 on a correct prediction, so only mistakes update.

    Stops after an epoch with zero updates, or after max_epochs.

    Pocket: after every update the accuracy on the full dataset is computed;
    if it beats the best seen so far, (w, b) is copied into the pocket.
    On separable data the pocket simply ends equal to the final weights.
    """
    w = np.array(w0, dtype=float).copy()
    b = float(b0)

    # Pocket starts with the initial weights.
    pocket_w, pocket_b = w.copy(), b
    pocket_acc = accuracy(X, y, w, b)
    pocket_epoch = 0

    history = {"acc": [], "pocket_acc": [], "updates": []}
    converged = False

    for epoch in range(1, max_epochs + 1):
        n_updates = 0
        for xi, yi in zip(X, y):
            y_hat = 1 if (xi @ w + b) >= 0 else 0  # same rule as step()
            error = yi - y_hat                    # 0, +1 or -1
            if error != 0:
                w += eta * error * xi
                b += eta * error
                n_updates += 1
                # Pocket bookkeeping: the only addition to the loop.
                acc_now = accuracy(X, y, w, b)
                if acc_now > pocket_acc:
                    pocket_w, pocket_b = w.copy(), b
                    pocket_acc = acc_now
                    pocket_epoch = epoch

        history["acc"].append(accuracy(X, y, w, b))
        history["pocket_acc"].append(pocket_acc)
        history["updates"].append(n_updates)

        if n_updates == 0:  # a full clean pass: converged
            converged = True
            break

    return {
        "w": w, "b": b, "acc": accuracy(X, y, w, b),
        "epochs": epoch, "converged": converged,
        "pocket_w": pocket_w, "pocket_b": pocket_b,
        "pocket_acc": pocket_acc, "pocket_epoch": pocket_epoch,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Plotting helpers (every plot: title, axis labels, class legend)
# ---------------------------------------------------------------------------
COLORS = {0: "#1f77b4", 1: "#ff7f0e"}


def scatter_classes(ax, X, y):
    for c in (0, 1):
        m = y == c
        ax.scatter(X[m, 0], X[m, 1], s=8, alpha=0.5, c=COLORS[c],
                   label=f"Class {c}")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")


def draw_boundary(ax, w, b, X, **kw):
    """Draw w1*x1 + w2*x2 + b = 0 across the x-range of the data."""
    x1 = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200)
    if abs(w[1]) > 1e-12:
        ax.plot(x1, -(w[0] * x1 + b) / w[1], **kw)
    else:  # vertical line
        ax.axvline(-b / w[0], **kw)


def mark_errors(ax, X, y, w, b, marker="x", color="red", label="Misclassified"):
    wrong = predict(X, w, b) != y
    if wrong.any():
        ax.scatter(X[wrong, 0], X[wrong, 1], marker=marker, s=30,
                   c=color, linewidths=1, label=f"{label} ({wrong.sum()})")
    return int(wrong.sum())


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Exercise 1 -- separable data
# ---------------------------------------------------------------------------
X1, y1 = make_dataset([1.5, 1.5], [[0.5, 0], [0, 0.5]],
                      [5.0, 5.0], [[0.5, 0], [0, 0.5]])

fig, ax = plt.subplots(figsize=(6, 5))
scatter_classes(ax, X1, y1)
ax.set_title("Figure 1 - Exercise 1: separable data (2000 points)")
ax.legend()
save(fig, "fig1.png")

# Initialization: w ~ N(0, 0.01), b = 0. Drawn ONCE and reused for the
# eta = 1.0 re-run, so that eta is the only thing that changes.
w_init_1 = rng.normal(0, 0.01, size=2)

r1 = train_perceptron(X1, y1, w_init_1, eta=0.01)
r1_eta1 = train_perceptron(X1, y1, w_init_1, eta=1.0)

fig, ax = plt.subplots(figsize=(6, 5))
scatter_classes(ax, X1, y1)
draw_boundary(ax, r1["w"], r1["b"], X1, color="black", lw=2,
              label="Decision boundary ($\\eta$=0.01)")
n_err_1 = mark_errors(ax, X1, y1, r1["w"], r1["b"])
ax.set_ylim(X1[:, 1].min() - 1, X1[:, 1].max() + 1)
ax.set_title("Figure 2 - Exercise 1: learned decision boundary")
ax.legend(loc="upper left")
save(fig, "fig2.png")

fig, ax = plt.subplots(figsize=(6, 4))
# epoch 0 = accuracy of the initial weights, before any training
acc0_1 = accuracy(X1, y1, w_init_1, 0.0)
ax.plot(np.arange(0, r1["epochs"] + 1), [acc0_1] + r1["history"]["acc"],
        "o-", label="$\\eta$ = 0.01")
ax.plot(np.arange(0, r1_eta1["epochs"] + 1), [acc0_1] + r1_eta1["history"]["acc"],
        "s--", label="$\\eta$ = 1.0 (item D)")
ax.set_xticks(np.arange(0, max(r1["epochs"], r1_eta1["epochs"]) + 1))
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy on the full dataset")
ax.set_title("Figure 3 - Exercise 1: accuracy x epoch")
ax.legend()
ax.grid(alpha=0.3)
save(fig, "fig3.png")


def unit(v):
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# Exercise 2 -- overlapping data
# ---------------------------------------------------------------------------
X2, y2 = make_dataset([3.0, 3.0], [[1.5, 0], [0, 1.5]],
                      [4.0, 4.0], [[1.5, 0], [0, 1.5]])

fig, ax = plt.subplots(figsize=(6, 5))
scatter_classes(ax, X2, y2)
ax.set_title("Figure 4 - Exercise 2: overlapping data (2000 points)")
ax.legend()
save(fig, "fig4.png")

w_init_2 = rng.normal(0, 0.01, size=2)
r2 = train_perceptron(X2, y2, w_init_2, eta=ETA)  # same function, unchanged

fig, ax = plt.subplots(figsize=(7, 5.5))
scatter_classes(ax, X2, y2)
draw_boundary(ax, r2["w"], r2["b"], X2, color="red", lw=2, ls="--",
              label="Final boundary")
draw_boundary(ax, r2["pocket_w"], r2["pocket_b"], X2, color="black", lw=2,
              label="Pocket boundary")
mark_errors(ax, X2, y2, r2["pocket_w"], r2["pocket_b"], marker="x",
            color="black", label="Misclassified by pocket")
mark_errors(ax, X2, y2, r2["w"], r2["b"], marker="o", color="none",
            label="Misclassified by final")
# hollow red circles for the final-weight errors
for coll in ax.collections[-1:]:
    coll.set_edgecolor("red")
    coll.set_facecolor("none")
ax.set_xlim(X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5)
ax.set_ylim(X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5)
ax.set_title("Figure 5 - Exercise 2: final vs pocket boundary")
ax.legend(loc="upper left", fontsize=8)
save(fig, "fig5.png")

fig, ax = plt.subplots(figsize=(7, 4))
ep2 = np.arange(1, r2["epochs"] + 1)
ax.plot(ep2, r2["history"]["acc"], lw=1, label="Current weights")
ax.plot(ep2, r2["history"]["pocket_acc"], lw=2, label="Pocket (best so far)")
ax.axhline(0.5, color="gray", ls=":", label="Chance (50%)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy on the full dataset")
ax.set_title("Figure 6 - Exercise 2: accuracy x epoch")
ax.legend()
ax.grid(alpha=0.3)
save(fig, "fig6.png")

# Where does the final boundary cut the diagonal x1 = x2 = t ?
# (w1 + w2) t + b = 0  ->  t = -b / (w1 + w2)
t_final = -r2["b"] / r2["w"].sum()
t_pocket = -r2["pocket_b"] / r2["pocket_w"].sum()
pred_final = predict(X2, r2["w"], r2["b"])

# ---------------------------------------------------------------------------
# Supplementary checks quoted in the D items (they draw nothing from rng)
# ---------------------------------------------------------------------------
# Ex1 D3: from a zero start, eta only rescales the weights.
z_a = train_perceptron(X1, y1, np.zeros(2), eta=0.01)
z_b = train_perceptron(X1, y1, np.zeros(2), eta=1.0)
zero_start = {
    "epochs_eta001": z_a["epochs"], "epochs_eta1": z_b["epochs"],
    "w_ratio": (z_b["w"] / z_a["w"]).tolist(),
    "b_ratio": z_b["b"] / z_a["b"],
}

# Ex2 D1: best accuracy any straight line reaches on THIS sample.
# Sweep directions theta and, for each, every threshold along the projection.
best_line_acc = 0.0
for theta in np.linspace(0, np.pi, 721):
    d = np.array([np.cos(theta), np.sin(theta)])
    proj = X2 @ d
    order = np.argsort(proj)
    ys = y2[order]
    # predict 1 to the right of the cut: correct = (#0 left) + (#1 right)
    zeros_left = np.concatenate([[0], np.cumsum(ys == 0)])
    ones_right = np.concatenate([[0], np.cumsum((ys == 1)[::-1])])[::-1]
    best_line_acc = max(best_line_acc, ((zeros_left + ones_right).max()) / len(ys))

# Ex2 D3: more epochs / smaller eta (same data, same initial weights).
more_epochs = train_perceptron(X2, y2, w_init_2, eta=0.01, max_epochs=500)
small_eta = train_perceptron(X2, y2, w_init_2, eta=0.001, max_epochs=100)
supp_ex2 = {
    "best_line_acc": float(best_line_acc),
    "epochs500_final_acc": more_epochs["acc"],
    "epochs500_pocket_acc": more_epochs["pocket_acc"],
    "epochs500_last100_acc_range": [min(more_epochs["history"]["acc"][-100:]),
                                    max(more_epochs["history"]["acc"][-100:])],
    "eta0001_final_acc": small_eta["acc"],
    "eta0001_pocket_acc": small_eta["pocket_acc"],
    "eta0001_acc_range": [min(small_eta["history"]["acc"]),
                          max(small_eta["history"]["acc"])],
}

# ---------------------------------------------------------------------------
# Print and save every reported number
# ---------------------------------------------------------------------------
results = {
    "ex1": {
        "w_init": w_init_1.tolist(),
        "acc_epoch0": acc0_1,
        "w": r1["w"].tolist(), "b": r1["b"], "epochs": r1["epochs"],
        "converged": r1["converged"], "acc": r1["acc"],
        "updates_per_epoch": r1["history"]["updates"],
        "acc_per_epoch": r1["history"]["acc"],
        "dir": unit(r1["w"]).tolist(),
        "misclassified": n_err_1,
    },
    "ex1_eta1": {
        "w": r1_eta1["w"].tolist(), "b": r1_eta1["b"],
        "epochs": r1_eta1["epochs"], "converged": r1_eta1["converged"],
        "acc": r1_eta1["acc"],
        "updates_per_epoch": r1_eta1["history"]["updates"],
        "dir": unit(r1_eta1["w"]).tolist(),
        "angle_deg": float(np.degrees(np.arccos(
            np.clip(unit(r1["w"]) @ unit(r1_eta1["w"]), -1, 1)))),
    },
    "ex2": {
        "w_init": w_init_2.tolist(),
        "w": r2["w"].tolist(), "b": r2["b"], "acc": r2["acc"],
        "epochs": r2["epochs"], "converged": r2["converged"],
        "pocket_w": r2["pocket_w"].tolist(), "pocket_b": r2["pocket_b"],
        "pocket_acc": r2["pocket_acc"], "pocket_epoch": r2["pocket_epoch"],
        "updates_per_epoch": r2["history"]["updates"],
        "acc_per_epoch": r2["history"]["acc"],
        "acc_min": min(r2["history"]["acc"]),
        "acc_max": max(r2["history"]["acc"]),
        "final_pred_frac_class1": float(pred_final.mean()),
        "diag_cut_final": float(t_final),
        "diag_cut_pocket": float(t_pocket),
        "mean_norm_x": float(np.linalg.norm(X2, axis=1).mean()),
    },
    "ex1_zero_start": zero_start,
    "ex2_supplementary": supp_ex2,
}

with open(HERE / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
