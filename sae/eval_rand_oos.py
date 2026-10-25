# Fix for eval_sae.py's random-init baseline: its least-squares decoder was fit ON the 10k held-out tokens it is scored on
# (3072x768 free parameters on 10k rows -> optimistic). Here: fit the LS decoder on 60k training tokens, score on the 10k held-out.
# Run: <venv>/bin/python -I eval_rand_oos.py <acts dir> <sae.pt>
import sys, glob, numpy as np, torch
d, sp = sys.argv[1], sys.argv[2]
torch.manual_seed(0); S = torch.load(sp, weights_only=False)
X = np.concatenate([np.load(f, mmap_mode="r") for f in sorted(glob.glob(d + "/acts_*.npy"))]).astype(np.float32)
X = (X - S["mu"]) / S["sc"]; perm = np.random.default_rng(0).permutation(len(X))
H = torch.tensor(X[perm[:10000]]); Tr = torch.tensor(X[perm[10000:70000]]); del X
tot = ((H - H.mean(0)) ** 2).sum(); fvu = lambda xh: float(((H - xh) ** 2).sum() / tot)
f = torch.relu((H - S["bd"]) @ S["We"].T + S["be"]); L0 = float((f > 0).float().sum(1).mean()); k = max(1, round(L0))
print(f"trained SAE: FVU {fvu(f @ S['Wd'].T + S['bd']):.4f} L0 {L0:.1f}")
for seed in range(3):
    torch.manual_seed(seed); Wd0 = torch.randn(768, S["Wd"].shape[1]); Wd0 /= Wd0.norm(dim=0, keepdim=True)
    def codes(A):
        a = torch.relu(A @ Wd0); v, i = torch.topk(a, k, dim=1); return torch.zeros_like(a).scatter(1, i, v)
    Ftr = codes(Tr); D = torch.linalg.lstsq(torch.cat([Ftr, torch.ones(len(Ftr), 1)], 1), Tr).solution
    Fh = codes(H); xh = torch.cat([Fh, torch.ones(len(Fh), 1)], 1) @ D
    print(f"random-init TopK k={k} seed {seed}: LS decoder fit on 60k train -> held-out FVU {fvu(xh):.4f}")
