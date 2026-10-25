# SAE (Bricken et al. 2023 recipe, as in sae.py) in torch on the streamed Pythia-160m layer-4 activations.
# f = ReLU(W_e(x − b_d) + b_e), x̂ = W_d f + b_d, loss = ||x − x̂||² + λ||f||₁, unit-norm decoder columns,
# dead-latent resampling every 1000 steps (first 60% of training). Activations centred + scaled to unit mean norm.
# Reports FVU (fraction of variance unexplained), L0, dead and dense fractions on 10k held-out tokens.
# Run: <venv>/bin/python -I sae_torch.py <acts dir> <width> <lambda> <steps> <out.pt>
import sys, glob, numpy as np, torch
d, F, lam, STEPS, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
torch.manual_seed(0); torch.set_num_threads(8)
X = np.concatenate([np.load(f, mmap_mode="r") for f in sorted(glob.glob(d + "/acts_*.npy"))]).astype(np.float32)
mu = X.mean(0); X -= mu; sc = np.linalg.norm(X, axis=1).mean(); X /= sc
rng = np.random.default_rng(0); perm = rng.permutation(len(X)); hold = torch.tensor(X[perm[:10000]]); tr = torch.tensor(X[perm[10000:]])
D = X.shape[1]; del X
Wd = torch.randn(D, F); Wd /= Wd.norm(dim=0, keepdim=True)
We = Wd.T.clone().requires_grad_(); be = torch.zeros(F, requires_grad=True); bd = torch.zeros(D, requires_grad=True); Wd.requires_grad_()
opt = torch.optim.Adam([We, be, Wd, bd], lr=1e-3); fired = torch.zeros(F, dtype=torch.bool)
def enc(x): return torch.relu((x - bd) @ We.T + be)
for t in range(1, STEPS + 1):
    x = tr[torch.randint(0, len(tr), (512,))]; f = enc(x); xh = f @ Wd.T + bd
    loss = ((xh - x) ** 2).sum(1).mean() + lam * f.abs().sum(1).mean()
    opt.zero_grad(); loss.backward()
    with torch.no_grad(): Wd.grad -= Wd * (Wd.grad * Wd).sum(0, keepdim=True)
    opt.step()
    with torch.no_grad():
        Wd /= Wd.norm(dim=0, keepdim=True); fired |= (f > 0).any(0)
        if t % 1000 == 0 and t <= 0.6 * STEPS:
            dead = (~fired).nonzero().squeeze(1)
            if len(dead):
                xs = tr[torch.randint(0, len(tr), (4096,))]; err = ((enc(xs) @ Wd.T + bd - xs) ** 2).sum(1)
                pick = xs[torch.multinomial(err, len(dead), replacement=True)]; v = pick / pick.norm(dim=1, keepdim=True)
                Wd[:, dead] = v.T; alive = fired.nonzero().squeeze(1)
                We[dead] = v * 0.2 * (We[alive].norm(dim=1).mean() if len(alive) else 1.0); be[dead] = 0
                opt = torch.optim.Adam([We, be, Wd, bd], lr=1e-3)
            fired[:] = False
    if t % 2000 == 0 or t == STEPS:
        with torch.no_grad():
            f = enc(hold); r = hold - (f @ Wd.T + bd); fvu = (r ** 2).sum() / ((hold - hold.mean(0)) ** 2).sum()
            dens = (f > 0).float().mean(0)
            print(f"step {t}: FVU {fvu:.4f} L0 {(f > 0).float().sum(1).mean():.1f} dead {(dens == 0).float().mean():.3f} dense>10% {(dens > 0.1).float().mean():.3f}", flush=True)
torch.save({"We": We.detach(), "be": be.detach(), "Wd": Wd.detach(), "bd": bd.detach(), "mu": mu, "sc": sc}, out)
