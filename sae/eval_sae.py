# Cheap evaluation of a trained SAE (plan from knowledge/sae-evaluation.md §6, checks 1–3, without the LM splice).
# Fidelity on the 10k held-out tokens used in training (same split, seed 0) against two baselines at matched L0:
#   random-init SAE (same init as sae_torch.py, TopK with k = trained L0) and PCA with rank = trained L0.
# Census: dense (>10%), antipodal decoder pairs (cos < −0.9), near-duplicates (cos > 0.9), shrinkage γ = E‖x̂‖²/E[x̂·x] (Gated eq. 10).
# Token identity (Heap 2025): per alive latent, share of activation mass on its top token id, trained vs random-init.
# Run: <venv>/bin/python -I eval_sae.py <acts dir> <sae.pt>
import sys, glob, numpy as np, torch
d, sp = sys.argv[1], sys.argv[2]
torch.manual_seed(0); S = torch.load(sp, weights_only=False)
X = np.concatenate([np.load(f, mmap_mode="r") for f in sorted(glob.glob(d + "/acts_*.npy"))]).astype(np.float32)
T = np.concatenate([np.load(f) for f in sorted(glob.glob(d + "/toks_*.npy"))])
X = (X - S["mu"]) / S["sc"]  # sae_torch.py computed mu, sc on all tokens before the split; reproduce its split exactly
perm = np.random.default_rng(0).permutation(len(X)); H = torch.tensor(X[perm[:10000]]); Ht = T[perm[:10000]]
tot = ((H - H.mean(0)) ** 2).sum()
def fvu(xh): return float(((H - xh) ** 2).sum() / tot)
f = torch.relu((H - S["bd"]) @ S["We"].T + S["be"]); xh = f @ S["Wd"].T + S["bd"]; L0 = float((f > 0).float().sum(1).mean()); k = max(1, round(L0))
print(f"trained: FVU {fvu(xh):.4f}  L0 {L0:.1f}  shrinkage γ {float((xh ** 2).sum(1).mean() / (xh * H).sum(1).mean()):.3f}")
Wd0 = torch.randn(768, S["Wd"].shape[1]); Wd0 /= Wd0.norm(dim=0, keepdim=True); We0 = Wd0.T
a0 = (H @ We0.T); v, i = torch.topk(torch.relu(a0), k, dim=1); f0 = torch.zeros_like(a0).scatter(1, i, v)
lsq = torch.linalg.lstsq(f0, H).solution  # best possible linear decoder for the random codes (generous to the baseline)
print(f"random-init SAE, TopK k={k}: FVU {fvu(f0 @ Wd0.T):.4f} (own decoder), {fvu(f0 @ lsq):.4f} (least-squares decoder)")
U, s, Vt = torch.linalg.svd(H - H.mean(0), full_matrices=False); P = Vt[:k]
print(f"PCA rank {k}: FVU {fvu((H - H.mean(0)) @ P.T @ P + H.mean(0)):.4f}")
dens = (f > 0).float().mean(0); alive = dens > 0
C = (S["Wd"].T @ S["Wd"]); C.fill_diagonal_(0)
print(f"census on held-out: alive {alive.float().mean():.3f}  dense>10% {(dens > 0.1).float().mean():.4f}  antipodal pairs {int((C < -0.9).sum()) // 2}  near-duplicate pairs {int((C > 0.9).sum()) // 2}")
def topshare(F):
    out = []
    for j in (F > 0).any(0).nonzero().squeeze(1).tolist():
        m = F[:, j].numpy(); s_ = np.bincount(Ht, weights=m); out.append(s_.max() / s_.sum())
    return np.array(out)
ts, ts0 = topshare(f), topshare(f0)
print(f"top-token mass share, median [IQR]: trained {np.median(ts):.2f} [{np.percentile(ts,25):.2f}, {np.percentile(ts,75):.2f}], share of latents >0.9: {np.mean(ts > 0.9):.2f}")
print(f"                                  random-init {np.median(ts0):.2f} [{np.percentile(ts0,25):.2f}, {np.percentile(ts0,75):.2f}], share >0.9: {np.mean(ts0 > 0.9):.2f}")
