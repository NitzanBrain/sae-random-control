# Robustness for token_eta.py (reviewer points): (1) cross-validated η² (token means from half A, scored on half B; unseen tokens
# predicted by the global mean) — removes the small-group upward bias; (2) η² on (previous token, current token) pairs, CV too;
# (3) η² on position in the window; (4) residual-level η²: share of total residual variance explained by token-id means (CV).
# All computed over ALL sampled tokens (zeros included). Run: <venv>/bin/python -I token_eta2.py <acts dir> <sae.pt> <label>
import sys, glob, numpy as np, torch
d, sp, lab = sys.argv[1:4]; S = torch.load(sp, weights_only=False)
T = np.concatenate([np.load(f) for f in sorted(glob.glob(d + "/toks_*.npy"))]); X = np.concatenate([np.load(f, mmap_mode="r") for f in sorted(glob.glob(d + "/acts_*.npy"))])
N = len(T); pos = np.arange(N) % 240
prev = np.concatenate([[-1], T[:-1]]); prev[pos == 0] = -1          # previous token inside the same window
idx = np.sort(np.random.default_rng(11).choice(N, 120000, replace=False))
H = (X[idx].astype(np.float32) - S["mu"]) / S["sc"]
F = torch.relu((torch.tensor(H) - S["bd"]) @ S["We"].T + S["be"]).numpy()
alive = (F > 0).sum(0) >= 40; F = F[:, alive]
A = np.random.default_rng(3).random(len(idx)) < 0.5
def cv_r2(V, key):
    # fit group means on A, predict B; R² on B relative to B's own variance
    ka, kb = key[A], key[~A]; u, inv = np.unique(ka, return_inverse=True); cnt = np.bincount(inv)
    sums = np.zeros((len(u),) + V.shape[1:]); np.add.at(sums, inv, V[A]); means = sums / cnt[:, None]
    g = V[A].mean(0); loc = np.searchsorted(u, kb); hit = (loc < len(u)) & (u[np.minimum(loc, len(u) - 1)] == kb)
    pred = np.where(hit[:, None], means[np.minimum(loc, len(u) - 1)], g)
    VB = V[~A]; return 1 - ((VB - pred) ** 2).sum(0) / ((VB - VB.mean(0)) ** 2).sum(0)
t = T[idx]; pair = prev[idx].astype(np.int64) * 60000 + t; p = pos[idx]
e_tok = cv_r2(F, t); e_pair = cv_r2(F, pair); e_pos = cv_r2(F, p)
dens = (F > 0).mean(0)
def s(e): return f"median {np.median(e):.2f}, share<0.5 {np.mean(e<0.5):.2f}"
print(f"[{lab}] alive {alive.sum()}")
print(f"  CV η² token      : {s(e_tok)}")
print(f"  CV η² prev+token : {s(e_pair)}   (latents with token<0.5 AND pair<0.5: {np.mean((e_tok<0.5)&(e_pair<0.5)):.2f})")
print(f"  CV η² position   : median {np.median(e_pos):.3f}, share>0.2 {np.mean(e_pos>0.2):.3f}")
for lo, hi in [(0, .002), (.002, .01), (.01, .05), (.05, 1)]:
    m = (dens >= lo) & (dens < hi)
    if m.sum(): print(f"  density [{lo},{hi}): n {m.sum()}, CV η² token median {np.median(e_tok[m]):.2f}, share<0.5 {np.mean(e_tok[m]<0.5):.2f}")
r = cv_r2(H, t); tot = ((H[~A] - H[~A].mean(0)) ** 2).sum(0)
print(f"  residual-level CV R² from token id (variance-weighted): {float((r * tot).sum() / tot.sum()):.3f}")
