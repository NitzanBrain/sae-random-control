# Rater-free "lexical vs contextual" measure: for each alive latent, η² = share of its activation variance (over 60k tokens)
# explained by the current token id (between-token variance / total). η²≈1: a pure token detector; low η²: depends on context.
# Run: <venv>/bin/python -I token_eta.py <acts dir> <sae.pt> [label]
import sys, glob, numpy as np, torch
d, sp = sys.argv[1], sys.argv[2]; lab = sys.argv[3] if len(sys.argv) > 3 else ""
S = torch.load(sp, weights_only=False)
T = np.concatenate([np.load(f) for f in sorted(glob.glob(d + "/toks_*.npy"))]); X = np.concatenate([np.load(f, mmap_mode="r") for f in sorted(glob.glob(d + "/acts_*.npy"))])
idx = np.sort(np.random.default_rng(11).choice(len(T), 60000, replace=False)); t = T[idx]
H = torch.tensor((X[idx].astype(np.float32) - S["mu"]) / S["sc"]); F = torch.relu((H - S["bd"]) @ S["We"].T + S["be"]).numpy()
u, inv = np.unique(t, return_inverse=True); cnt = np.bincount(inv)
def eta(F):
    tot = ((F - F.mean(0)) ** 2).sum(0); sums = np.zeros((len(u), F.shape[1])); np.add.at(sums, inv, F)
    btw = (sums ** 2 / cnt[:, None]).sum(0) - F.sum(0) ** 2 / len(F); return btw / np.maximum(tot, 1e-12)
alive = (F > 0).sum(0) >= 20; e = eta(F[:, alive])
# chance level: η² from token ids alone is inflated by rare tokens; shuffle tokens to get the null
rng = np.random.default_rng(0); Fs = F[:, alive][rng.permutation(len(F))]; e0 = eta(Fs)
print(f"{lab} alive {alive.sum()}  η² median {np.median(e):.2f} [IQR {np.percentile(e,25):.2f}-{np.percentile(e,75):.2f}]  share η²<0.5: {np.mean(e<0.5):.2f}  share η²>0.9: {np.mean(e>0.9):.2f}  | shuffled-null median {np.median(e0):.2f}")
