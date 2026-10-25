# Blind interpretability set: 20 random alive trained latents + 20 random-init directions (TopK-22 codes, same as the baseline).
# For each: top-8 activating contexts on 50k held-out-ish tokens (from the training pool, but no labels). Shuffled; key saved separately.
# Run: <venv>/bin/python -I blind_dump.py <acts dir> <sae.pt> <tokenizer dir> <out dir>
import sys, glob, json, numpy as np, torch
from transformers import AutoTokenizer
d, sp, td, od = sys.argv[1:5]; tok = AutoTokenizer.from_pretrained(td); S = torch.load(sp, weights_only=False)
X = np.concatenate([np.load(f, mmap_mode="r") for f in sorted(glob.glob(d + "/acts_*.npy"))])
T = np.concatenate([np.load(f) for f in sorted(glob.glob(d + "/toks_*.npy"))])
rng = np.random.default_rng(7); idx = np.sort(rng.choice(len(X), 60000, replace=False))
H = torch.tensor((X[idx].astype(np.float32) - S["mu"]) / S["sc"])
f = torch.relu((H - S["bd"]) @ S["We"].T + S["be"])
torch.manual_seed(5); Wd0 = torch.randn(768, f.shape[1]); Wd0 /= Wd0.norm(dim=0, keepdim=True)
a = torch.relu(H @ Wd0); v, ix = torch.topk(a, 22, dim=1); f0 = torch.zeros_like(a).scatter(1, ix, v)
alive = ((f > 0).sum(0) >= 8).nonzero().squeeze(1).numpy(); alive0 = ((f0 > 0).sum(0) >= 8).nonzero().squeeze(1).numpy()
pick = [("T", int(j)) for j in rng.choice(alive, 20, replace=False)] + [("R", int(j)) for j in rng.choice(alive0, 20, replace=False)]
rng.shuffle(pick); items, key = [], {}
def ctx(p):
    g = idx[p]; s = max(0, g - 8); e = min(len(T), g + 4)
    return tok.decode(T[s:g].tolist()) + " [[" + tok.decode([int(T[g])]) + "]] " + tok.decode(T[g + 1:e].tolist())
for n, (src, j) in enumerate(pick):
    F = (f if src == "T" else f0)[:, j]; top = torch.topk(F, 8).indices.numpy()
    items.append({"id": n, "contexts": [ctx(p) for p in top]}); key[n] = [src, j]
json.dump(items, open(f"{od}/blind_items.json", "w"), indent=1, ensure_ascii=False); json.dump(key, open(f"{od}/blind_key.json", "w"))
print(len(items), "items")
