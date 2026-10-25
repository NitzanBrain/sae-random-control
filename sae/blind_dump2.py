# Blind set B (Heap-2025 control): 20 latents of the SAE on the TRAINED Pythia (label P) + 20 latents of the SAE on a
# RANDOM-INIT Pythia (label Z), same tokens, same 60k sample, shuffled. Key saved separately.
# Run: <venv>/bin/python -I blind_dump2.py <actsP> <saeP> <actsZ> <saeZ> <tokenizer dir> <out dir>
import sys, glob, json, numpy as np, torch
from transformers import AutoTokenizer
dP, sP, dZ, sZ, td, od = sys.argv[1:7]; tok = AutoTokenizer.from_pretrained(td)
T = np.concatenate([np.load(f) for f in sorted(glob.glob(dP + "/toks_*.npy"))])
assert (T == np.concatenate([np.load(f) for f in sorted(glob.glob(dZ + "/toks_*.npy"))])).all()
rng = np.random.default_rng(11); idx = np.sort(rng.choice(len(T), 60000, replace=False))
def codes(d, sp):
    S = torch.load(sp, weights_only=False); X = np.concatenate([np.load(f, mmap_mode="r") for f in sorted(glob.glob(d + "/acts_*.npy"))])
    H = torch.tensor((X[idx].astype(np.float32) - S["mu"]) / S["sc"]); return torch.relu((H - S["bd"]) @ S["We"].T + S["be"])
F = {"P": codes(dP, sP), "Z": codes(dZ, sZ)}
pick = []
for lab in "PZ":
    alive = ((F[lab] > 0).sum(0) >= 8).nonzero().squeeze(1).numpy(); pick += [(lab, int(j)) for j in rng.choice(alive, 20, replace=False)]
rng.shuffle(pick)
def ctx(p):
    g = idx[p]; s = max(0, g - 8); e = min(len(T), g + 4)
    return tok.decode(T[s:g].tolist()) + " [[" + tok.decode([int(T[g])]) + "]] " + tok.decode(T[g + 1:e].tolist())
items, key = [], {}
for n, (lab, j) in enumerate(pick):
    top = torch.topk(F[lab][:, j], 8).indices.numpy(); items.append({"id": n, "contexts": [ctx(p) for p in top]}); key[n] = [lab, j]
json.dump(items, open(f"{od}/blindB_items.json", "w"), indent=1, ensure_ascii=False); json.dump(key, open(f"{od}/blindB_key.json", "w")); print(len(items))
