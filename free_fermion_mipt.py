#!/usr/bin/env python3
"""
================================================================
  free_fermion_mipt_gpu.py
  GPU-accelerated Free-Fermion MIPT (PyTorch + CUDA)

  Usage:
    python free_fermion_mipt_gpu.py                 # full run
    python free_fermion_mipt_gpu.py --quick          # quick test
    python free_fermion_mipt_gpu.py --resume         # only run remaining α values
================================================================
"""

import sys, os, time, csv
import numpy as np
import torch

if torch.cuda.is_available():
    DEV = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    free_mem, total_mem = torch.cuda.mem_get_info(0)
    print(f"✓ PyTorch CUDA — {props.name}")
    print(f"  VRAM: {total_mem/1e9:.1f} GB total, {free_mem/1e9:.1f} GB free")
else:
    DEV = torch.device("cpu")
    free_mem = 2 * 1024**3
    total_mem = free_mem
    print("✗ No CUDA — running on CPU")

CDTYPE = torch.complex128
RDTYPE = torch.float64


def build_hamiltonian(L, alpha):
    idx = torch.arange(L, dtype=RDTYPE, device=DEV)
    dist = (idx[:, None] - idx[None, :]).abs()
    dist.fill_diagonal_(1.0)
    if alpha < 0.001:
        h = -torch.ones(L, L, dtype=RDTYPE, device=DEV)
        h.fill_diagonal_(0.0)
    else:
        h = -dist.pow(-alpha)
        h.fill_diagonal_(0.0)
    h /= (h.abs().sum() / L)
    return h


def build_unitary(h, dt):
    ev, V = torch.linalg.eigh(h)
    phases = torch.exp(-1j * ev.to(CDTYPE) * dt)
    return (V.to(CDTYPE) * phases.unsqueeze(0)) @ V.T.to(CDTYPE)


class BatchRunner:
    def __init__(self, L, U, gamma, dt, n_steps, batch, seed=42):
        self.L, self.batch = L, batch
        self.U, self.Ud = U, U.conj().T
        self.gamma, self.dt, self.n_steps = gamma, dt, n_steps
        self.p_step = min(gamma * dt, 1.0)
        self.rng = np.random.RandomState(seed)

    def init_C(self):
        B, L = self.batch, self.L
        C = torch.zeros(B, L, L, dtype=CDTYPE, device=DEV)
        for b in range(B):
            for s in self.rng.permutation(L)[:L//2]:
                C[b, s, s] = 1.0
        return C

    def evolve(self, C):
        return self.U @ C @ self.Ud

    def measure(self, C):
        B, L = self.batch, self.L
        for k in range(L):
            do_m = self.rng.random(B) < self.p_step
            if not do_m.any():
                continue
            idx = torch.tensor(np.where(do_m)[0], device=DEV, dtype=torch.long)
            n = idx.shape[0]
            p1 = C[idx, k, k].real.clamp(0.0, 1.0)
            rv = torch.tensor(self.rng.random(n), device=DEV, dtype=RDTYPE)
            is1 = rv < p1

            i1 = idx[is1]
            if i1.numel() > 0:
                ps = p1[is1]; safe = ps > 1e-12; i1s = i1[safe]; ps = ps[safe]
                if i1s.numel() > 0:
                    rk = C[i1s, k, :].clone(); ck = C[i1s, :, k].clone()
                    vi = -ck; vi[:, k] += 1.0
                    C[i1s] += torch.einsum('bi,bj->bij', vi, rk) / ps[:, None, None]

            i0 = idx[~is1]
            if i0.numel() > 0:
                p0 = 1.0 - p1[~is1]; safe = p0 > 1e-12; i0s = i0[safe]; p0s = p0[safe]
                if i0s.numel() > 0:
                    rk = C[i0s, k, :].clone(); ck = C[i0s, :, k].clone()
                    vi = -ck; vi[:, k] += 1.0
                    C[i0s] -= torch.einsum('bi,bj->bij', vi, rk) / p0s[:, None, None]
        return C

    def purify(self, C):
        C = 0.5 * (C + C.conj().transpose(-2, -1))
        Cr = C.real; Cr = 0.5 * (Cr + Cr.transpose(-2, -1))

        bad = torch.isnan(Cr).reshape(self.batch, -1).any(dim=1)
        if bad.any():
            for b in torch.where(bad)[0].tolist():
                Cr[b] = 0.0
                for s in self.rng.permutation(self.L)[:self.L//2]:
                    Cr[b, s, s] = 1.0

        w, V = self._safe_eigh_batch(Cr)
        w = w.clamp(0.0, 1.0)
        return ((V * w.unsqueeze(-2)) @ V.transpose(-2, -1)).to(CDTYPE)

    def _safe_eigh_batch(self, M):
        """eigh with per-element fallback on failure."""
        try:
            return torch.linalg.eigh(M)
        except torch.linalg.LinAlgError:
            B, N = M.shape[0], M.shape[1]
            w = torch.zeros(B, N, dtype=RDTYPE, device=DEV)
            V = torch.zeros(B, N, N, dtype=RDTYPE, device=DEV)
            for b in range(B):
                try:
                    w[b], V[b] = torch.linalg.eigh(M[b])
                except Exception:
                    V[b] = torch.eye(N, dtype=RDTYPE, device=DEV)
                    for s in self.rng.permutation(N)[:N//2]:
                        w[b, s] = 1.0
            return w, V

    def entropy(self, C):
        LA = self.L // 2
        CA = C[:, :LA, :LA]
        CA = 0.5 * (CA + CA.conj().transpose(-2, -1))
        CAr = 0.5 * (CA.real + CA.real.transpose(-2, -1))

        try:
            nu, _ = torch.linalg.eigh(CAr)
        except torch.linalg.LinAlgError:
            # per-trajectory fallback
            S = torch.zeros(self.batch, dtype=RDTYPE, device=DEV)
            for b in range(self.batch):
                try:
                    nub, _ = torch.linalg.eigh(CAr[b])
                    nub = nub.clamp(1e-14, 1.0 - 1e-14)
                    S[b] = -(nub * nub.log2() + (1-nub) * (1-nub).log2()).sum()
                except Exception:
                    S[b] = 0.0
            return S

        nu = nu.clamp(1e-14, 1.0 - 1e-14)
        return -(nu * nu.log2() + (1.0 - nu) * (1.0 - nu).log2()).sum(dim=-1)

    def run(self):
        C = self.init_C()
        for _ in range(self.n_steps // 4):
            C = self.evolve(C)
        for step in range(self.n_steps):
            C = self.evolve(C)
            C = self.measure(C)
            if step % 5 == 0:
                C = self.purify(C)
        return self.entropy(C).cpu().numpy()


def crossing_fraction(L, alpha):
    cut = L // 2; cross = total = 0.0
    for i in range(L):
        for j in range(i+1, L):
            w = 1.0 if alpha < 0.001 else (j-i)**(-alpha)
            total += w
            if i < cut <= j: cross += w
    return cross / total


def load_existing(path):
    """Load already-computed (alpha, gamma, L) combos from CSV."""
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for row in csv.DictReader(f):
                try:
                    a = float(row['alpha'])
                    g = float(row['gamma'])
                    L = int(row['L'])
                    done.add((round(a,2), round(g,2), L))
                except (ValueError, KeyError):
                    pass
    return done


def main():
    quick   = "--quick"  in sys.argv
    resume  = "--resume" in sys.argv

    os.makedirs("results", exist_ok=True)
    dt = 0.05
    csv_path = "results/free_fermion_mipt_gpu.csv"

    if quick:
        Ls = [32, 64, 128]
        alphas = [0.0, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0]
        gammas = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
        n_traj, n_steps = 20, 400
    else:
        Ls = [32, 64, 128, 256]
        alphas = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]
        gammas = [0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
        n_traj, n_steps = 30, 600

    # --resume: load existing results, skip what's done
    done = set()
    if resume:
        done = load_existing(csv_path)
        total_combos = len(alphas) * len(gammas) * len(Ls)
        print(f"  Resume mode: {len(done)}/{total_combos} combos already done, skipping them.")

    usable = int(free_mem * 0.6) if DEV.type == "cuda" else 2*1024**3

    print("="*64)
    print("  Free-Fermion MIPT Verification (GPU)")
    print(f"  dt = {dt:.3f}, Ls = {Ls}")
    print("  Minato bound: α_c = 3/2.  Kac prediction: α_c = 1.")
    if resume:
        remaining_a = sorted(set(a for a in alphas
                                 for g in gammas for L in Ls
                                 if (round(a,2), round(g,2), L) not in done))
        print(f"  Remaining α values: {remaining_a}")
    print("="*64 + "\n")

    # Open CSV in append mode if resuming, else write fresh
    if resume and os.path.exists(csv_path):
        fp = open(csv_path, "a")
    else:
        fp = open(csv_path, "w")
        fp.write("alpha,gamma,L,S_mean,S_err,S_over_Smax,f_cross,n_traj\n")

    t_global = time.time()

    for alpha in alphas:
        # Check if this entire alpha is done
        alpha_combos = [(round(alpha,2), round(g,2), L) for g in gammas for L in Ls]
        if resume and all(c in done for c in alpha_combos):
            print(f"--- α = {alpha:.1f} --- SKIP (all done)")
            continue

        print(f"--- α = {alpha:.1f} ---")

        for L in Ls:
            t0 = time.time()
            h = build_hamiltonian(L, alpha)
            U = build_unitary(h, dt)
            fc = crossing_fraction(L, alpha)

            mem_per = L*L*16*3
            max_batch = max(1, usable // mem_per)
            nt = n_traj
            if L >= 256: nt = max(5, n_traj//3)
            elif L >= 128: nt = max(10, n_traj*2//3)
            bs = min(nt, max_batch)

            if L == Ls[0]:
                print(f"  (build: {time.time()-t0:.1f}s, batch={bs})")

            for gamma in gammas:
                key = (round(alpha,2), round(gamma,2), L)
                if resume and key in done:
                    continue

                t0 = time.time()
                all_S, rem, soff = [], nt, 0
                while rem > 0:
                    b = min(rem, bs)
                    seed = 42 + int(alpha*100)*10000 + L*100000 + int(gamma*100)*1000000 + soff
                    all_S.extend(BatchRunner(L, U, gamma, dt, n_steps, b, seed).run().tolist())
                    rem -= b; soff += b

                all_S = np.array(all_S)
                S_m = all_S.mean()
                S_e = all_S.std(ddof=1)/np.sqrt(len(all_S)) if len(all_S)>1 else 0.0
                Smax = L/2.0; sr = S_m/Smax
                el = time.time()-t0

                fp.write(f"{alpha:.2f},{gamma:.2f},{L},{S_m:.4f},{S_e:.4f},{sr:.4f},{fc:.4f},{nt}\n")
                fp.flush()
                print(f"  L={L:3d} γ={gamma:5.1f}  S/Smax={sr:.3f} ± {S_e/Smax:.3f}  ({nt} traj, {el:.1f}s)")

            if DEV.type == "cuda": torch.cuda.empty_cache()
        print()

    fp.close()

    tot = time.time()-t_global
    print("="*64)
    print(f"  DONE. {tot:.0f}s ({tot/60:.1f} min)")
    print(f"  Output: {csv_path}")
    print("="*64)

    # ── β analysis on ALL data in CSV ──
    print(f"\n  β ANALYSIS (S ~ L^β):\n")
    data = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            data.append({
                'alpha': float(row['alpha']),
                'gamma': float(row['gamma']),
                'L': int(row['L']),
                'S_mean': float(row['S_mean'])
            })

    all_alphas = sorted(set(d['alpha'] for d in data))
    all_gammas = sorted(set(d['gamma'] for d in data))
    all_Ls = sorted(set(d['L'] for d in data))

    def fS(a, g, L):
        for d in data:
            if abs(d['alpha']-a)<0.01 and abs(d['gamma']-g)<0.01 and d['L']==L:
                return d['S_mean']
        return -1

    hdr = f"  {'α':5s} {'γ':6s}"
    for i in range(len(all_Ls)-1): hdr += f"  β({all_Ls[i]}→{all_Ls[i+1]})"
    hdr += "  avg    verdict"
    print(hdr); print("  "+"-"*len(hdr))

    for alpha in all_alphas:
        for gamma in all_gammas:
            line = f"  {alpha:<5.1f} {gamma:<6.1f}"
            betas = []
            for i in range(len(all_Ls)-1):
                S1, S2 = fS(alpha, gamma, all_Ls[i]), fS(alpha, gamma, all_Ls[i+1])
                if S1 > 0.01 and S2 > 0.01:
                    b = np.log(S2/S1)/np.log(all_Ls[i+1]/all_Ls[i]); betas.append(b)
                    line += f"  {b:8.2f}  "
                else:
                    line += f"  {'---':>8s}  "
            if betas:
                avg = np.mean(betas)
                v = "VOLUME" if avg>0.7 else ("sub-vol" if avg>0.3 else "AREA")
                line += f"  {avg:5.2f}  {v}"
                if alpha<1 and avg>0.5: line += "  ← α<1, volume ✓"
                if alpha>1 and avg<0.3: line += "  ← α>1, area ✓"
            print(line)

    print(f"\n  SUMMARY:")
    print("  α<1 VOLUME → α_c=1 universal | α≈1.5 → Minato | α≈1.0 → Kac")
    print("="*64)

if __name__ == "__main__":
    main()
