"""
Variational autoencoder on MNIST (PyTorch) -- Insper ANN-DL 2026.2, VAE exercise.

Run from the repository root:
    python docs/exercises/vae/code/vae.py

Trains the d = 16 VAE (Exercise 1) and the d = 2 VAE (Exercise 2), saves
Figures 1-6 to ../figures/ and writes every number the report quotes to
results.json. MNIST is downloaded by torchvision into $MNIST_ROOT
(default: /tmp/mnist), outside the repository.

Written by hand here: the reparameterization, both loss terms (BCE summed
over pixels, closed-form KL) and the sampling. No library VAE class and no
torch.distributions are used. Autograd computes the gradients.
"""

import json
import os
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)
MNIST_ROOT = os.environ.get("MNIST_ROOT", "/tmp/mnist")

torch.use_deterministic_algorithms(True)
torch.set_num_threads(2)          # fixed thread count, so sums are done the same way
R = {}

EPOCHS, BATCH, LR = 20, 128, 1e-3


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=130)
    plt.close(fig)


# ===========================================================================
# 1.A Data: ToTensor() scales to [0, 1]; images are flattened to 784-vectors
# ===========================================================================
def load(train):
    ds = datasets.MNIST(MNIST_ROOT, train=train, download=True, transform=transforms.ToTensor())
    X = torch.stack([img.view(-1) for img, _ in ds])          # (N, 784) in [0, 1]
    y = torch.tensor([lab for _, lab in ds])
    return TensorDataset(X, y)


full_train, test_set = load(True), load(False)
train_set, val_set = random_split(full_train, [50_000, 10_000],
                                  generator=torch.Generator().manual_seed(42))
Xte, yte = test_set.tensors
R["sizes"] = {"train": len(train_set), "val": len(val_set), "test": len(test_set)}


# ===========================================================================
# 1.B Model
# ===========================================================================
class VAE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        self.mu_head = nn.Linear(256, d)      # mean of q(z|x)
        self.logvar_head = nn.Linear(256, d)  # log sigma^2 of q(z|x)
        self.dec = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 512), nn.ReLU(),
                                 nn.Linear(512, 784), nn.Sigmoid())

    def encode(self, x):
        h = self.enc(x)
        return self.mu_head(h), self.logvar_head(h)

    def decode(self, z):
        return self.dec(z)

    @staticmethod
    def reparameterize(mu, logvar):
        """z = mu + sigma * eps, eps ~ N(0, I): the noise is an input, so
        gradients flow to mu and logvar through a deterministic function."""
        eps = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar) * eps

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_terms(x_hat, x, mu, logvar):
    """Both terms in nats per image (sum over pixels / latent dims, mean over batch)."""
    n = x.shape[0]
    recon = F.binary_cross_entropy(x_hat, x, reduction="sum") / n
    # KL(N(mu, sigma^2) || N(0, 1)) in closed form, summed over latent dimensions
    kl = 0.5 * torch.sum(mu ** 2 + logvar.exp() - 1 - logvar) / n
    return recon, kl


@torch.no_grad()
def evaluate(model, dataset):
    """Average reconstruction and KL (nats/image) over a whole dataset."""
    model.eval()
    rec_sum, kl_sum, n = 0.0, 0.0, 0
    for x, _ in DataLoader(dataset, batch_size=1000):
        x_hat, mu, logvar = model(x)
        rec, kl = loss_terms(x_hat, x, mu, logvar)
        rec_sum += rec.item() * len(x); kl_sum += kl.item() * len(x); n += len(x)
    model.train()
    return rec_sum / n, kl_sum / n


# ===========================================================================
# 1.C Training (same function for d = 16 and d = 2)
# ===========================================================================
def train_vae(d):
    torch.manual_seed(42)                       # before building each model
    model = VAE(d)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loader = DataLoader(train_set, batch_size=BATCH, shuffle=True)
    hist = {"train_rec": [], "train_kl": [], "val_rec": [], "val_kl": []}
    for epoch in range(EPOCHS):
        for x, _ in loader:
            x_hat, mu, logvar = model(x)
            rec, kl = loss_terms(x_hat, x, mu, logvar)
            loss = rec + kl                     # beta = 1: negative ELBO
            opt.zero_grad()
            loss.backward()
            opt.step()
        # end of epoch: both terms on the full train and validation sets
        for split, ds in (("train", train_set), ("val", val_set)):
            r, k = evaluate(model, ds)
            hist[f"{split}_rec"].append(r); hist[f"{split}_kl"].append(k)
        print(f"d={d} epoch {epoch + 1:2d}  train rec {hist['train_rec'][-1]:.2f} kl "
              f"{hist['train_kl'][-1]:.2f} | val rec {hist['val_rec'][-1]:.2f} kl {hist['val_kl'][-1]:.2f}")
    return model, hist


vae16, hist16 = train_vae(16)
torch.manual_seed(42)                           # fixes the eps used for the test numbers
rec16, kl16 = evaluate(vae16, test_set)        # test set used only here, at the end
R["ex1"] = {"history": hist16, "test_rec": rec16, "test_kl": kl16, "test_neg_elbo": rec16 + kl16,
            "val_kl_first": hist16["val_kl"][0], "val_kl_last": hist16["val_kl"][-1],
            "val_rec_first": hist16["val_rec"][0], "val_rec_last": hist16["val_rec"][-1]}

ep = np.arange(1, EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, term, label in [(axes[0], "rec", "reconstruction (BCE)"), (axes[1], "kl", "KL")]:
    ax.plot(ep, hist16[f"train_{term}"], "o-", ms=3, label="train")
    ax.plot(ep, hist16[f"val_{term}"], "s-", ms=3, label="validation")
    ax.set(title=f"{label} vs. epoch (d = 16)", xlabel="epoch", ylabel="nats per image")
    ax.legend()
fig.suptitle("Figure 1. Exercise 1: the two loss terms during training")
save(fig, "fig1.png")


# ===========================================================================
# 1.D Reconstructions (decode mu) and prior samples
# ===========================================================================
first_idx = [int((yte == k).nonzero()[0]) for k in range(10)]
R["ex1"]["first_test_index_per_digit"] = first_idx
vae16.eval()
with torch.no_grad():
    x = Xte[first_idx]
    mu, _ = vae16.encode(x)
    x_rec = vae16.decode(mu)
fig, axes = plt.subplots(2, 10, figsize=(12, 2.8))
for k in range(10):
    axes[0, k].imshow(x[k].view(28, 28), cmap="gray"); axes[0, k].set_title(f"digit {k}", fontsize=8)
    axes[1, k].imshow(x_rec[k].view(28, 28), cmap="gray")
for a in axes.ravel():
    a.set_xticks([]); a.set_yticks([])
axes[0, 0].set_ylabel("original"); axes[1, 0].set_ylabel("decode(μ)")
fig.suptitle("Figure 2. First test image of each digit (top) and its reconstruction (bottom), d = 16")
save(fig, "fig2.png")

torch.manual_seed(0)
with torch.no_grad():
    samples = vae16.decode(torch.randn(64, 16))
fig, axes = plt.subplots(8, 8, figsize=(7, 7.4))
for a, img in zip(axes.ravel(), samples):
    a.imshow(img.view(28, 28), cmap="gray"); a.set_xticks([]); a.set_yticks([])
fig.suptitle("Figure 3. 64 samples: decode(z), z ~ N(0, I), d = 16")
save(fig, "fig3.png")

# Supporting numbers for 1.E: how far prior samples are from where the
# encoder actually puts test images, and how grey the decoder outputs are.
with torch.no_grad():
    mu_te, logvar_te = vae16.encode(Xte)
    z_prior = torch.randn(10_000, 16, generator=torch.Generator().manual_seed(0))
    d_prior = torch.cdist(z_prior, mu_te).min(1).values        # nearest encoded test image
    d_test = torch.cdist(mu_te[:5000], mu_te[5000:]).min(1).values
    out_prior = vae16.decode(z_prior)
    out_mu = vae16.decode(mu_te)
grey = lambda o: float(((o > 0.2) & (o < 0.8)).float().mean())   # share of "undecided" pixels
R["ex1"]["latent_stats"] = {
    "mean_sigma_test": float(torch.exp(0.5 * logvar_te).mean()),
    "mean_sigma_per_dim": torch.exp(0.5 * logvar_te).mean(0).tolist(),
    "aggregate_mu_std_per_dim": torch.std(mu_te, 0).tolist(),
    "median_dist_prior_to_nearest_mu": float(d_prior.median()),
    "median_dist_mu_to_nearest_mu": float(d_test.median()),
    "grey_pixel_share_prior_samples": grey(out_prior),
    "grey_pixel_share_decoded_mu": grey(out_mu),
    "grey_pixel_share_real_test": grey(Xte),
}


# ===========================================================================
# Exercise 2 -- 2-D latent space
# ===========================================================================
vae2, hist2 = train_vae(2)
torch.manual_seed(42)
rec2, kl2 = evaluate(vae2, test_set)
R["ex2"] = {"history": hist2, "test_rec": rec2, "test_kl": kl2, "test_neg_elbo": rec2 + kl2}

vae2.eval()
with torch.no_grad():
    mu2, _ = vae2.encode(Xte)
mu2 = mu2.numpy()
fig, ax = plt.subplots(figsize=(8, 7))
cmap = plt.get_cmap("tab10")
for k in range(10):
    m = yte.numpy() == k
    ax.scatter(mu2[m, 0], mu2[m, 1], s=3, alpha=0.6, color=cmap(k), label=str(k))
ax.set(title="Figure 4. μ of all 10,000 test images, d = 2", xlabel="z1 (μ₁)", ylabel="z2 (μ₂)")
ax.legend(title="digit", markerscale=4, loc="best", fontsize=8)
save(fig, "fig4.png")
centroids = {k: mu2[yte.numpy() == k].mean(0).tolist() for k in range(10)}
R["ex2"]["centroids"] = centroids
# overlap: for each digit, which other digit's centroid is closest, and the
# share of test points whose nearest centroid is another digit
C = np.array([centroids[k] for k in range(10)])
nearest_c = np.linalg.norm(mu2[:, None, :] - C[None], axis=2).argmin(1)
conf = np.zeros((10, 10), int)
for t, p in zip(yte.numpy(), nearest_c):
    conf[t, p] += 1
pairs = sorted(((conf[i, j] + conf[j, i], i, j) for i in range(10) for j in range(i + 1, 10)), reverse=True)
R["ex2"]["centroid_confusion_top_pairs"] = [[int(n), int(i), int(j)] for n, i, j in pairs[:5]]

q = np.linspace(0.05, 0.95, 15)
grid = norm.ppf(q)
R["ex2"]["grid_values"] = grid.tolist()
zz = torch.tensor([[gx, gy] for gy in grid[::-1] for gx in grid], dtype=torch.float32)
with torch.no_grad():
    imgs = vae2.decode(zz).view(15, 15, 28, 28).numpy()
canvas = np.ones((15 * 28, 15 * 28))
for r in range(15):
    for c in range(15):
        canvas[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = imgs[r, c]
fig, ax = plt.subplots(figsize=(8.5, 8.5))
ax.imshow(canvas, cmap="gray", extent=[-0.5, 14.5, 14.5, -0.5])
ticks = list(range(0, 15, 2))
ax.set_xticks(ticks); ax.set_xticklabels([f"{grid[i]:.2f}" for i in ticks], fontsize=7)
ax.set_yticks(ticks); ax.set_yticklabels([f"{grid[::-1][i]:.2f}" for i in ticks], fontsize=7)
ax.set(title="Figure 5. Decoded 15×15 grid, z = Φ⁻¹(q), q in [0.05, 0.95], d = 2",
       xlabel="z1", ylabel="z2")
save(fig, "fig5.png")

# 2.D Interpolation between the first test "1" and the first test "0"
ia, ib = first_idx[1], first_idx[0]
R["ex2"]["interp_indices"] = {"a": ia, "a_digit": int(yte[ia]), "b": ib, "b_digit": int(yte[ib])}
ts = torch.linspace(0, 1, 10)
xa, xb = Xte[ia], Xte[ib]
pixel_row = torch.stack([(1 - t) * xa + t * xb for t in ts])
with torch.no_grad():
    ma, _ = vae16.encode(xa[None]); mb, _ = vae16.encode(xb[None])
    latent_row = vae16.decode(torch.cat([(1 - t) * ma + t * mb for t in ts]))
fig, axes = plt.subplots(2, 10, figsize=(12, 2.9))
for k in range(10):
    axes[0, k].imshow(pixel_row[k].view(28, 28), cmap="gray")
    axes[1, k].imshow(latent_row[k].view(28, 28), cmap="gray")
    axes[0, k].set_title(f"t = {ts[k]:.2f}", fontsize=8)
for a in axes.ravel():
    a.set_xticks([]); a.set_yticks([])
axes[0, 0].set_ylabel("pixel"); axes[1, 0].set_ylabel("latent (d=16)")
fig.suptitle(f"Figure 6. Test image {ia} (a {int(yte[ia])}) to test image {ib} (a {int(yte[ib])})")
save(fig, "fig6.png")

with open(HERE / "results.json", "w") as f:
    json.dump(R, f, indent=2)
print(json.dumps({k: v for k, v in R.items()}, indent=1)[:3000])
