"""
Data preparation and analysis -- Insper ANN-DL 2026.2, Exercise 1 (Data).

Run from the repository root:
    python docs/exercises/data/code/data.py

It builds every dataset, prints and saves every number the report quotes
(results.json), and saves Figures 1-7 to ../figures/.
A single generator, rng = np.random.default_rng(42), is used for everything
(Exercise 1 noise, Exercise 2 samples, the Exercise 3 split), so every run
produces exactly the same numbers and figures. No model is trained.
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # no display needed, we only save PNGs
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE.parent / "figures"
DATA_CSV = HERE.parent / "data" / "train.csv"
FIG_DIR.mkdir(exist_ok=True)

# The ONLY random generator in the whole script.
rng = np.random.default_rng(42)

R = {}  # every number quoted in the report ends up here -> results.json
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=130)
    plt.close(fig)


# ===========================================================================
# Exercise 1 -- point clouds in 2D
# ===========================================================================
MEANS = np.array([[2, 3], [5, 6], [8, 1], [15, 4]], dtype=float)
STDS = np.array([[0.8, 2.5], [1.2, 1.9], [0.9, 0.9], [0.5, 2.0]])
N_PER_CLASS = 100
SCALES = [0.5, 1.0, 2.0, 4.0]

# A. Draw the standard-normal noise ONCE (100 x 2 per class). A dataset with
# spread s is mean + s * std * Z, so the four datasets of part B differ only
# in spread, not in which random points were drawn, and s = 1 is exactly the
# Figure 1 dataset.
Z = rng.standard_normal((4, N_PER_CLASS, 2))
y1 = np.repeat(np.arange(4), N_PER_CLASS)


def clouds(s):
    """400 x 2 points with all standard deviations multiplied by s."""
    return np.vstack([MEANS[k] + s * STDS[k] * Z[k] for k in range(4)])


X1 = clouds(1.0)

fig, ax = plt.subplots(figsize=(7, 5))
for k in range(4):
    ax.scatter(*X1[y1 == k].T, s=14, alpha=0.7, c=COLORS[k], label=f"Class {k}")
    ax.scatter(*MEANS[k], marker="X", s=180, c=COLORS[k], edgecolors="black")
ax.scatter([], [], marker="X", s=120, c="white", edgecolors="black", label="Class mean")
ax.set(title="Figure 1. Four Gaussian classes (s = 1)", xlabel="x1", ylabel="x2")
ax.legend()
save(fig, "fig1.png")

# Figure 1b (extra, item C.2): the boundaries an ideal classifier would draw,
# computed from the KNOWN generating Gaussians (Bayes rule, equal priors).
# Nothing is fitted; it is the target a trained network would approximate.
gx, gy = np.meshgrid(np.linspace(-3, 19, 500), np.linspace(-6, 12, 400))
G = np.c_[gx.ravel(), gy.ravel()]
loglik = np.stack([
    -np.sum(((G - MEANS[k]) / STDS[k]) ** 2, axis=1) / 2 - np.log(STDS[k]).sum()
    for k in range(4)
])
region = loglik.argmax(0).reshape(gx.shape)
fig, ax = plt.subplots(figsize=(7, 5))
ax.contourf(gx, gy, region, levels=[-0.5, 0.5, 1.5, 2.5, 3.5], colors=COLORS, alpha=0.15)
ax.contour(gx, gy, region, levels=[0.5, 1.5, 2.5], colors="black", linewidths=1)
for k in range(4):
    ax.scatter(*X1[y1 == k].T, s=12, alpha=0.7, c=COLORS[k], label=f"Class {k}")
ax.set(title="Figure 1b. Ideal (Bayes) decision regions from the true Gaussians",
       xlabel="x1", ylabel="x2")
ax.legend()
save(fig, "fig1b.png")

# B.1 Figure 2: four spreads, shared axis limits.
datasets = {s: clouds(s) for s in SCALES}
allpts = np.vstack(list(datasets.values()))
lims = (allpts.min(0) - 1, allpts.max(0) + 1)
fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
for ax, s in zip(axes.ravel(), SCALES):
    for k in range(4):
        ax.scatter(*datasets[s][y1 == k].T, s=8, alpha=0.6, c=COLORS[k], label=f"Class {k}")
    ax.set(title=f"s = {s}", xlabel="x1", ylabel="x2", xlim=(lims[0][0], lims[1][0]),
           ylim=(lims[0][1], lims[1][1]))
axes[0, 0].legend(fontsize=8)
fig.suptitle("Figure 2. Same means, standard deviations multiplied by s")
save(fig, "fig2.png")

# B.2 Separation ratio r_ij = ||mu_i - mu_j|| / (sbar_i + sbar_j), at s = 1.
sbar = STDS.mean(1)
pairs = []
for i, j in combinations(range(4), 2):
    dist = float(np.linalg.norm(MEANS[i] - MEANS[j]))
    r = dist / (sbar[i] + sbar[j])
    pairs.append({"pair": f"{i}-{j}", "dist": dist, "r_s1": r, "r_s2": r / 2})
smallest = min(pairs, key=lambda p: p["r_s1"])
R["ex1_sbar"] = sbar.tolist()
R["ex1_pairs"] = pairs
R["ex1_smallest_pair"] = smallest

# B.3 Mixing rate: fraction of points whose nearest TRUE class mean is not
# their own class mean.
def mixing_rate(X):
    d = np.linalg.norm(X[:, None, :] - MEANS[None, :, :], axis=2)
    return float((d.argmin(1) != y1).mean())

mix = {s: mixing_rate(datasets[s]) for s in SCALES}
R["ex1_mixing"] = {str(s): mix[s] for s in SCALES}
# per-pair confusion at each s (which classes get mixed)
conf = {}
for s in SCALES:
    d = np.linalg.norm(datasets[s][:, None, :] - MEANS[None], axis=2).argmin(1)
    conf[str(s)] = pd.crosstab(y1, d).reindex(index=range(4), columns=range(4), fill_value=0).values.tolist()
R["ex1_confusion_nearest_mean"] = conf

# Exact linear-separability check (no training): a linear programme asks
# whether weights W (4 x 2) and biases b (4) exist with
#   w_y . x + b_y >= w_k . x + b_k + 1   for every point x of class y, k != y.
# Feasible <=> a set of straight boundaries (a linear layer + argmax, i.e.
# convex polygonal regions) puts every point in its own class region.
def linearly_separable(X, y, n_cls):
    d = X.shape[1] + 1                    # weights + bias per class
    Xb = np.c_[X, np.ones(len(X))]
    rows = []
    for x, c in zip(Xb, y):
        for k in range(n_cls):
            if k == c:
                continue
            row = np.zeros(n_cls * d)
            row[c * d:(c + 1) * d] = -x   # -(w_c.x) + (w_k.x) <= -1
            row[k * d:(k + 1) * d] = x
            rows.append(row)
    res = linprog(np.zeros(n_cls * d), A_ub=np.array(rows), b_ub=-np.ones(len(rows)),
                  bounds=[(None, None)] * (n_cls * d), method="highs")
    return res.status == 0                # 0 = feasible, 2 = infeasible


def pairwise_separable(X):
    out = {}
    for i, j in combinations(range(4), 2):
        m = (y1 == i) | (y1 == j)
        out[f"{i}-{j}"] = linearly_separable(X[m], (y1[m] == j).astype(int), 2)
    return out


R["ex1_linsep_all4"] = {str(s): linearly_separable(datasets[s], y1, 4) for s in SCALES}
R["ex1_linsep_pairs"] = {str(s): pairwise_separable(datasets[s]) for s in SCALES}

# Figure 3 on a fine grid of s (same Z), so the onset of mixing is visible.
s_grid = np.round(np.arange(0.25, 4.01, 0.05), 2)
mix_grid = [mixing_rate(clouds(s)) for s in s_grid]
first_mix = float(s_grid[np.argmax(np.array(mix_grid) > 0)])
sep_grid = [linearly_separable(clouds(s), y1, 4) for s in s_grid]
last_sep = float(s_grid[np.flatnonzero(sep_grid).max()])
first_nonsep = float(s_grid[np.argmin(sep_grid)])
R["ex1_linsep_grid"] = {"last_separable_s": last_sep, "first_nonseparable_s": first_nonsep,
                        "monotone": bool(all(sep_grid[:np.argmin(sep_grid)]) and not any(sep_grid[np.argmin(sep_grid):]))}
R["ex1_smallest_r_at_first_nonsep"] = smallest["r_s1"] / first_nonsep
R["ex1_first_s_with_mixing"] = first_mix
R["ex1_smallest_r_at_first_mix"] = smallest["r_s1"] / first_mix
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(s_grid, mix_grid, c="gray", lw=1, label="fine grid of s")
ax.plot(SCALES, [mix[s] for s in SCALES], "o-", c="black", label="the 4 datasets")
ax.axvline(first_nonsep, ls="--", c="tab:red",
           label=f"no longer linearly separable, s = {first_nonsep}")
ax.set(title="Figure 3. Mixing rate vs. spread s", xlabel="s (std multiplier)",
       ylabel="mixing rate (nearest mean is wrong)")
ax.legend()
save(fig, "fig3.png")


# ===========================================================================
# Exercise 2 -- non-linearity in 5D
# ===========================================================================
N2 = 500
mu_A = np.zeros(5)
cov_A = np.array([
    [1.0, 0.8, 0.1, 0.0, 0.0],
    [0.8, 1.0, 0.3, 0.0, 0.0],
    [0.1, 0.3, 1.0, 0.5, 0.0],
    [0.0, 0.0, 0.5, 1.0, 0.2],
    [0.0, 0.0, 0.0, 0.2, 1.0],
])
mu_B = np.full(5, 1.5)
cov_B = np.array([
    [1.5, -0.7, 0.2, 0.0, 0.0],
    [-0.7, 1.5, 0.4, 0.0, 0.0],
    [0.2, 0.4, 1.5, 0.6, 0.0],
    [0.0, 0.0, 0.6, 1.5, 0.3],
    [0.0, 0.0, 0.0, 0.3, 1.5],
])
# Both matrices must be valid covariances (symmetric, positive definite).
R["ex2_min_eig"] = {"A": float(np.linalg.eigvalsh(cov_A).min()),
                    "B": float(np.linalg.eigvalsh(cov_B).min())}

# A. Dataset I
XA = rng.multivariate_normal(mu_A, cov_A, size=N2)
XB = rng.multivariate_normal(mu_B, cov_B, size=N2)
X_I = np.vstack([XA, XB])
y_I = np.repeat([0, 1], N2)


# B. Dataset II: random direction on the unit sphere times a random radius.
def shell(mean_radius, n=N2):
    v = rng.standard_normal((n, 5))
    u = v / np.linalg.norm(v, axis=1, keepdims=True)
    rho = rng.normal(mean_radius, 0.4, size=n)
    return rho[:, None] * u

XC = shell(2.0)
XD = shell(5.0)
X_II = np.vstack([XC, XD])
y_II = np.repeat([0, 1], N2)

# C.1 / C.2 PCA to 2D, one PCA per dataset (fitted on both classes together).
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (key, name, X, y, labels) in zip(axes, [
    ("I", "Dataset I (shifted Gaussians)", X_I, y_I, ["Class A", "Class B"]),
    ("II", "Dataset II (concentric shells)", X_II, y_II, ["Class C (core)", "Class D (shell)"]),
]):
    pca = PCA(n_components=2).fit(X)
    P = pca.transform(X)
    ev = pca.explained_variance_ratio_
    R[f"ex2_pca_{key}"] = {"pc1": float(ev[0]), "pc2": float(ev[1]), "sum": float(ev.sum())}
    R[f"ex2_pca_{key}"]["abs_cos_pc1_vs_ones"] = float(abs(pca.components_[0] @ (np.ones(5) / np.sqrt(5))))
    for c in (0, 1):
        ax.scatter(*P[y == c].T, s=8, alpha=0.6, c=COLORS[c], label=labels[c])
    ax.set(title=f"{name}\nPC1 {ev[0]:.1%}, PC2 {ev[1]:.1%}", xlabel="PC1", ylabel="PC2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend()
fig.suptitle("Figure 4. PCA projections to 2D")
save(fig, "fig4.png")

# C.3 Distance between the (sample) class centers in 5D, and radius stats.
R["ex2_center_dist_I"] = float(np.linalg.norm(XA.mean(0) - XB.mean(0)))
R["ex2_center_dist_I_true"] = float(np.linalg.norm(mu_A - mu_B))
R["ex2_center_dist_II"] = float(np.linalg.norm(XC.mean(0) - XD.mean(0)))
norms = {k: np.linalg.norm(v, axis=1) for k, v in
         {"A": XA, "B": XB, "C": XC, "D": XD}.items()}
R["ex2_norm_stats"] = {k: {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}
                       for k, v in norms.items()}

# D.3 A simple separating function for Dataset II: f(x) = ||x||^2 - t.
# t is placed halfway (in radius) between the two mean radii: (2 + 5)/2 = 3.5.
t = 3.5 ** 2
pred_II = (np.sum(X_II ** 2, axis=1) > t).astype(int)
R["ex2_rule_threshold_sq"] = t
R["ex2_rule_accuracy"] = float((pred_II == y_II).mean())
R["ex2_gap"] = {"max_core_norm": float(norms["C"].max()), "min_shell_norm": float(norms["D"].min())}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
bins = np.linspace(0, max(n.max() for n in norms.values()) + 0.5, 50)
for ax, (title, ks, labels) in zip(axes, [
    ("Dataset I", ["A", "B"], ["Class A", "Class B"]),
    ("Dataset II", ["C", "D"], ["Class C (core)", "Class D (shell)"]),
]):
    for c, (k, lab) in enumerate(zip(ks, labels)):
        ax.hist(norms[k], bins=bins, alpha=0.55, color=COLORS[c], label=lab)
    ax.set(title=title, xlabel="||x|| (5D Euclidean norm)", ylabel="count")
    ax.legend()
axes[1].axvline(3.5, ls="--", c="black", label="||x|| = 3.5")
axes[1].legend()
fig.suptitle("Figure 5. Distribution of ||x|| per class")
save(fig, "fig5.png")


# ===========================================================================
# Exercise 3 -- Spaceship Titanic
# ===========================================================================
SPEND = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
CAT_ONEHOT = ["HomePlanet", "CryoSleep", "Destination", "VIP"]

df = pd.read_csv(DATA_CSV)
R["ex3_shape_raw"] = list(df.shape)

# A.2 class balance
bal = df["Transported"].value_counts()
R["ex3_balance"] = {"True": int(bal[True]), "False": int(bal[False]),
                    "share_true": float(bal[True] / len(df))}

# A.4 missing-value table
miss = pd.DataFrame({"missing": df.isna().sum(), "pct": 100 * df.isna().mean()})
R["ex3_missing"] = {c: [int(r.missing), round(float(r.pct), 2)] for c, r in miss.iterrows()}
R["ex3_rows_with_any_nan"] = int(df.isna().any(axis=1).sum())

# A.5 spending stats (full raw file, NaN ignored)
R["ex3_spend_stats_full"] = {
    c: {"mean": float(df[c].mean()), "median": float(df[c].median()),
        "max": float(df[c].max()), "zero_share": float((df[c] == 0).mean())}
    for c in SPEND
}

# B. Stratified 80/20 split using the SAME rng: inside each class, permute
# the row indices and send the first 20% to test.
test_idx = []
for label in (False, True):
    idx = np.flatnonzero(df["Transported"].values == label)
    idx = rng.permutation(idx)
    test_idx.extend(idx[: int(round(0.2 * len(idx)))])
test_mask = np.zeros(len(df), dtype=bool)
test_mask[test_idx] = True
train, test = df[~test_mask].copy(), df[test_mask].copy()
R["ex3_split"] = {"train": len(train), "test": len(test),
                  "share_true_train": float(train["Transported"].mean()),
                  "share_true_test": float(test["Transported"].mean())}
R["ex3_foodcourt_train_raw"] = {"mean": float(train["FoodCourt"].mean()),
                                "median": float(train["FoodCourt"].median())}
foodcourt_raw_train = train["FoodCourt"].dropna().values  # for Figure 6
vrdeck_raw_train = train["VRDeck"].dropna().values          # for Figure 7

# C.1 Missing data. All statistics come from TRAIN only.
#  - spending: a CryoSleep passenger is confined to the cabin and spends 0
#    (checked below on train), so NaN -> 0 when CryoSleep is True; otherwise
#    NaN -> train median of that column.
#  - Age: train median (robust to the long right tail of ages).
#  - categorical: train mode.
R["ex3_cryo_spend_check_train"] = float(
    train.loc[train["CryoSleep"] == True, SPEND].fillna(0).sum(axis=1).max()
)
fit = {
    "age_median": float(train["Age"].median()),
    "spend_median": {c: float(train[c].median()) for c in SPEND},
    "mode": {c: train[c].mode()[0] for c in CAT_ONEHOT},
}


def impute(d):
    d = d.copy()
    for c in CAT_ONEHOT:
        d[c] = d[c].fillna(fit["mode"][c])
    asleep = d["CryoSleep"] == True
    for c in SPEND:
        d.loc[asleep & d[c].isna(), c] = 0.0
        d[c] = d[c].fillna(fit["spend_median"][c])
    d["Age"] = d["Age"].fillna(fit["age_median"])
    return d


train, test = impute(train), impute(test)
R["ex3_impute_fit"] = {"age_median": fit["age_median"], "spend_median": fit["spend_median"],
                       "mode": {k: str(v) for k, v in fit["mode"].items()}}

# C.3 Feature engineering (after imputation, so no NaN propagates).
for d in (train, test):
    d["TotalSpend"] = d[SPEND].sum(axis=1)
    d.drop(columns=["Cabin", "Name", "PassengerId"], inplace=True)

# C.2 One-hot encoding, fitted on train. handle_unknown="ignore": a category
# never seen in train becomes an all-zero row in that feature's block.
for d in (train, test):
    for c in CAT_ONEHOT:
        d[c] = d[c].astype(str)
enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(train[CAT_ONEHOT])
onehot_names = list(enc.get_feature_names_out(CAT_ONEHOT))
demo = pd.DataFrame([{"HomePlanet": "Pluto", "CryoSleep": "False",
                      "Destination": "TRAPPIST-1e", "VIP": "False"}])
demo_row = enc.transform(demo)[0]
R["ex3_unknown_category_demo"] = {"input": demo.iloc[0].to_dict(),
                                  "HomePlanet_block": demo_row[:3].tolist()}

# C.4 Heavy tails: log(1 + x) on the five spending columns and TotalSpend.
LOG_COLS = SPEND + ["TotalSpend"]
for d in (train, test):
    d[LOG_COLS] = np.log1p(d[LOG_COLS])

# C.5 Standardization (z-score), mean and std from TRAIN only.
NUM = ["Age"] + LOG_COLS
mu, sd = train[NUM].mean(), train[NUM].std(ddof=0)


def build_matrix(d):
    num = ((d[NUM] - mu) / sd).values
    cat = enc.transform(d[CAT_ONEHOT])
    return np.hstack([num, cat])


Xtr, Xte = build_matrix(train), build_matrix(test)
ytr = train["Transported"].astype(int).values
yte = test["Transported"].astype(int).values
feature_names = NUM + onehot_names
R["ex3_features"] = feature_names
R["ex3_shape_train"] = list(Xtr.shape)
R["ex3_shape_test"] = list(Xte.shape)
R["ex3_nan_left"] = {"train": int(np.isnan(Xtr).sum()), "test": int(np.isnan(Xte).sum())}
R["ex3_range_numeric"] = {
    "train": [float(Xtr[:, : len(NUM)].min()), float(Xtr[:, : len(NUM)].max())],
    "test": [float(Xte[:, : len(NUM)].min()), float(Xte[:, : len(NUM)].max())],
}
R["ex3_range_full"] = {"train": [float(Xtr.min()), float(Xtr.max())],
                       "test": [float(Xte.min()), float(Xte.max())]}
R["ex3_per_column_range_train"] = {
    n: [round(float(Xtr[:, i].min()), 3), round(float(Xtr[:, i].max()), 3)]
    for i, n in enumerate(NUM)
}
R["ex3_train_mean_std_after"] = {
    n: [round(float(Xtr[:, i].mean()), 6), round(float(Xtr[:, i].std()), 6)]
    for i, n in enumerate(NUM)
}
# Share of |z| > 2 on the numeric block (where tanh starts to saturate).
R["ex3_share_abs_gt2_train"] = float((np.abs(Xtr[:, : len(NUM)]) > 2).mean())

# Figure 6: FoodCourt, raw train vs. final (log + z-score).
fc_final = Xtr[:, NUM.index("FoodCourt")]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].hist(foodcourt_raw_train, bins=60, color="tab:blue", label="train, raw")
axes[0].set(title="Before: raw FoodCourt", xlabel="FoodCourt spend (credits)", ylabel="passengers")
axes[1].hist(fc_final, bins=60, color="tab:green", label="train, log1p + z-score")
axes[1].set(title="After: log(1+x), then standardized", xlabel="FoodCourt (z-score of log1p)",
            ylabel="passengers")
for ax in axes:
    ax.set_yscale("log")  # same (log) count axis on both panels
    ax.legend()
fig.suptitle("Figure 6. FoodCourt before and after preprocessing (training set)")
save(fig, "fig6.png")

# Figure 7 (item C.4): VRDeck before vs. after log(1 + x) only.
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].hist(vrdeck_raw_train, bins=60, color="tab:red", label="train, raw")
axes[0].set(title="Before: raw VRDeck", xlabel="VRDeck spend (credits)", ylabel="passengers")
axes[1].hist(np.log1p(vrdeck_raw_train), bins=60, color="tab:purple", label="train, log1p")
axes[1].set(title="After: log(1 + VRDeck)", xlabel="log(1 + VRDeck)", ylabel="passengers")
for ax in axes:
    ax.legend()
fig.suptitle("Figure 7. Effect of log(1 + x) on a spending column (training set)")
save(fig, "fig7.png")

R["ex3_raw_skew_train"] = {c: float(pd.Series(foodcourt_raw_train).skew()) for c in ["FoodCourt"]}
R["ex3_log_skew_train"] = {"FoodCourt": float(pd.Series(np.log1p(foodcourt_raw_train)).skew())}

# ---------------------------------------------------------------------------
with open(HERE / "results.json", "w") as f:
    json.dump(R, f, indent=2, default=str)
print(json.dumps(R, indent=2, default=str))
