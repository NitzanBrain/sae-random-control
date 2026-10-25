# Look at what SAE latents respond to: for random alive latents, the top-activating tokens with ±6 tokens of context.
# Also a crude interpretability check: share of a latent's top-20 activations whose token is the same string ("token-identity" latents).
# Run: <venv>/bin/python -I inspect_feats.py <acts dir> <sae.pt> <tokenizer dir> <n latents>
import sys, glob, numpy as np, torch
from transformers import AutoTokenizer
d, sp, td, N = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
tok = AutoTokenizer.from_pretrained(td); S = torch.load(sp, weights_only=False)
X = np.concatenate([np.load(f, mmap_mode="r") for f in sorted(glob.glob(d + "/acts_*.npy"))]).astype(np.float32)
T = np.concatenate([np.load(f) for f in sorted(glob.glob(d + "/toks_*.npy"))])
X = torch.tensor((X - S["mu"]) / S["sc"])
enc = lambda x, W=S["We"], b=S["be"]: torch.relu((x - S["bd"]) @ W.T + b)
cnt = torch.zeros(S["We"].shape[0])
for k in range(0, len(X), 8192): cnt += (enc(X[k:k + 8192]) > 0).float().sum(0)  # chunked: the full matrix would be ~2.4 GB
dens = cnt / len(X); alive = (dens > 0).nonzero().squeeze(1).numpy()
rng = np.random.default_rng(1); pick = rng.choice(alive, min(N, len(alive)), replace=False)
Fp = torch.cat([enc(X[k:k + 8192], S["We"][pick], S["be"][pick]) for k in range(0, len(X), 8192)])
col = {j: c for c, j in enumerate(pick)}
same = []
for j in pick:
    top = torch.topk(Fp[:, col[j]], 20).indices.numpy(); strs = [tok.decode([T[i]]) for i in top]
    same.append(max(strs.count(s) for s in set(strs)) / 20)
    print(f"\n### latent {j}  density {dens[j]:.4%}  top-token share {same[-1]:.2f}")
    for i in top[:6]:
        w = i % 240; a = max(i - min(w, 6), 0)  # stay inside the window
        ctx = tok.decode(T[a:i]) + " [[" + tok.decode([T[i]]) + "]] " + tok.decode(T[i + 1:min(i + 4, i - w + 240)])
        print(f"  {Fp[i, col[j]]:.2f}  {ctx!r}")
print(f"\nmean top-token share over {len(pick)} latents: {np.mean(same):.2f}; latents with share ≥0.9: {np.mean(np.array(same) >= 0.9):.2f}")
