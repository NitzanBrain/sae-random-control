# Memory-light extraction (the full version was killed twice for low RAM): same as extract_hf.py, but writes each
# book's activations to its own .npy as it goes (float16), never concatenating in memory, and fewer windows.
# Run: HF_HOME=$TMPDIR/hf <venv>/bin/python -I extract_stream.py <model dir> <out dir> <layer> <windows per book> books...
import sys, re, os, numpy as np, torch
md, out, LAYER, NWB, books = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5:]
from transformers import GPTNeoXForCausalLM, AutoTokenizer
torch.set_num_threads(8)
tok = AutoTokenizer.from_pretrained(md); hf = GPTNeoXForCausalLM.from_pretrained(md, torch_dtype=torch.float32).eval()
hf.gpt_neox.layers = hf.gpt_neox.layers[:LAYER + 2]  # keep one extra block: HF applies the final LayerNorm to the LAST hidden state
def clean(t):
    t = t.replace("\r\n", "\n"); t = re.sub(r"[ \t]+", " ", t); t = re.sub(r"\n\s*\n+", "\n\n", t)
    i = t.find("*** START"); j = t.find("*** END"); return t[(t.find("\n", i) if i >= 0 else 0):(j if j > 0 else None)]
os.makedirs(out, exist_ok=True)
for bi, bk in enumerate(books):
    ids = tok(clean(open(bk, encoding="utf-8").read()))["input_ids"]; starts = np.linspace(1000, len(ids) - 1257, NWB).astype(int)
    A = np.lib.format.open_memmap(f"{out}/acts_{bi}.npy", mode="w+", dtype=np.float16, shape=(NWB * 240, 768)); T = np.zeros(NWB * 240, np.int32)
    for k in range(0, NWB, 8):
        b = torch.tensor(np.stack([ids[s:s + 256] for s in starts[k:k + 8]]))
        with torch.no_grad(): h = hf(b, output_hidden_states=True).hidden_states[LAYER + 1][:, 16:]
        n = h.shape[0] * 240; A[k * 240:k * 240 + n] = h.reshape(-1, 768).numpy().astype(np.float16); T[k * 240:k * 240 + n] = b[:, 16:].reshape(-1).numpy()
    A.flush(); del A; np.save(f"{out}/toks_{bi}.npy", T); print(bk, "done", NWB * 240, "tokens", flush=True)
