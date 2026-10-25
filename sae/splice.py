# LM-splice test: replace Pythia-160m's layer-4 residual (output of block 4) by the SAE reconstruction and measure next-token loss
# on an UNSEEN book (Alice in Wonderland), against clean, mean-ablation, PCA-22 and random-init TopK (train-fit LS decoder) splices.
# Run: <venv>/bin/python -I splice.py <model dir> <sae.pt> <acts dir> <book>
import sys, re, glob, numpy as np, torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer
md, sp, d, bk = sys.argv[1:5]; torch.set_num_threads(8); torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(md); hf = GPTNeoXForCausalLM.from_pretrained(md, torch_dtype=torch.float32).eval()
S = torch.load(sp, weights_only=False); mu = torch.tensor(S["mu"]); sc = float(S["sc"])
t = open(bk, encoding="utf-8").read(); i = t.find("*** START"); j = t.find("*** END"); t = re.sub(r"\s+", " ", t[t.find("\n", i):j])
ids = tok(t)["input_ids"]; starts = np.linspace(0, len(ids) - 257, 96).astype(int)
B = torch.tensor(np.stack([ids[s:s + 256] for s in starts]))
# baselines fitted on training activations
X = np.concatenate([np.load(f, mmap_mode="r") for f in sorted(glob.glob(d + "/acts_*.npy"))]).astype(np.float32)
X = (X - S["mu"]) / sc; Tr = torch.tensor(X[np.random.default_rng(1).permutation(len(X))[:60000]]); del X
k = 22; U, s_, Vt = torch.linalg.svd(Tr - Tr.mean(0), full_matrices=False); P = Vt[:k]; m0 = Tr.mean(0)
Wd0 = torch.randn(768, S["Wd"].shape[1]); Wd0 /= Wd0.norm(dim=0, keepdim=True)
def rcodes(A): a = torch.relu(A @ Wd0); v, ix = torch.topk(a, k, dim=1); return torch.zeros_like(a).scatter(1, ix, v)
D0 = torch.linalg.lstsq(torch.cat([rcodes(Tr), torch.ones(len(Tr), 1)], 1), Tr).solution
def sae(x): f = torch.relu((x - S["bd"]) @ S["We"].T + S["be"]); return f @ S["Wd"].T + S["bd"]
modes = {"clean": None, "sae": sae, "pca22": lambda x: (x - m0) @ P.T @ P + m0,
         "randTopK_LS": lambda x: torch.cat([rcodes(x), torch.ones(len(x), 1)], 1) @ D0, "mean_ablate": lambda x: m0.expand_as(x)}
cur = {"f": None}
def hook(mod, inp, out):
    if cur["f"] is None: return out
    h = out[0] if isinstance(out, tuple) else out; sh = h.shape
    x = (h.reshape(-1, 768) - mu) / sc; h2 = (cur["f"](x) * sc + mu).reshape(sh)
    h2[:, 0] = h[:, 0]   # keep position 0 (attention-sink token; SAEs are usually not trained on it)
    return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
hf.gpt_neox.layers[4].register_forward_hook(hook)
res = {}
for name, fn in modes.items():
    cur["f"] = fn; L = []
    with torch.no_grad():
        for b in range(0, len(B), 8):
            x = B[b:b + 8]; lg = hf(x).logits[:, 16:-1]; L.append(torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), x[:, 17:].reshape(-1), reduction="none"))
    res[name] = torch.cat(L).mean().item(); print(f"{name:12s} loss {res[name]:.4f}", flush=True)
c, m = res["clean"], res["mean_ablate"]
for n in ["sae", "pca22", "randTopK_LS"]: print(f"{n}: loss recovered {(m - res[n]) / (m - c):.3f}  (Δ vs clean {res[n] - c:+.4f})")
