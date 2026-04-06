"""
Haar-random gate verification of α_c = 1 at small system sizes.
REVISION: numpy vectorized for ~100x speedup; added L=16; more trajectories.

Uses exact state-vector simulation (no Clifford tricks) with Haar-random
2-qubit gates at L = 8, 10, 12, 14, 16 for α = 0, 0.5, 1.0, 1.5, 2.0.

Output: results/haar_check.csv
"""

import numpy as np
from scipy.stats import unitary_group
import os, time

# ============================================================
# Vectorized exact state-vector simulation
# ============================================================

def random_haar_2qubit():
    """Sample a Haar-random 4x4 unitary."""
    return unitary_group.rvs(4)

def apply_two_qubit_gate(psi, L, q1, q2, U):
    """Apply 4x4 unitary U to qubits q1, q2 — vectorized via reshape+transpose.

    Convention: psi is reshaped as a tensor with shape [2]*L, where axis k
    corresponds to qubit k.  Qubits q1,q2 are moved to the last two axes,
    giving a (batch, 4) matrix that U acts on from the right as T @ U.T.
    Since U is Haar-random, any fixed basis ordering produces the same
    ensemble-averaged results.
    """
    shape = [2] * L
    T = psi.reshape(shape)
    # Move q1, q2 to the last two axes
    axes = list(range(L))
    axes.remove(q1)
    axes.remove(q2)
    axes.extend([q1, q2])
    T = np.transpose(T, axes)
    # Reshape: merge all other axes, leave last 2 as (2,2) → flatten to 4
    batch_shape = T.shape[:-2]
    T = T.reshape(-1, 4)  # (batch, 4)
    T = T @ U.T            # (batch, 4)
    T = T.reshape(batch_shape + (2, 2))
    # Inverse transpose
    inv_axes = [0] * L
    for i, a in enumerate(axes):
        inv_axes[a] = i
    T = np.transpose(T, inv_axes)
    return T.reshape(-1)

def measure_z(psi, L, q):
    """Projective Z-measurement on qubit q — vectorized."""
    shape = [2] * L
    T = psi.reshape(shape)
    # Probability of outcome 0: sum |psi|^2 over the slice where qubit q = 0
    slc0 = [slice(None)] * L
    slc0[q] = 0
    p0 = np.sum(np.abs(T[tuple(slc0)])**2)

    outcome = 0 if np.random.random() < p0 else 1
    # Project: zero out the other outcome
    slc_kill = [slice(None)] * L
    slc_kill[q] = 1 - outcome
    T_new = T.copy()
    T_new[tuple(slc_kill)] = 0.0
    psi_new = T_new.reshape(-1)
    norm = np.linalg.norm(psi_new)
    if norm > 1e-15:
        psi_new /= norm
    return outcome, psi_new

def half_chain_entropy(psi, L):
    """Von Neumann entropy of the half-chain reduced density matrix."""
    cut = L // 2
    dim_A = 1 << cut
    dim_B = 1 << (L - cut)
    # Reshape: A = qubits 0..cut-1 (low bits), B = qubits cut..L-1 (high bits)
    psi_mat = psi.reshape(dim_B, dim_A).T  # (dim_A, dim_B)
    s = np.linalg.svd(psi_mat, compute_uv=False)
    s = s[s > 1e-15]
    return -np.sum(s**2 * np.log2(s**2))

# ============================================================
# Power-law bond weights
# ============================================================

def make_bonds(L, alpha):
    bonds = []
    weights = []
    for i in range(L):
        for j in range(i+1, L):
            d = j - i
            w = d**(-alpha) if alpha > 0.001 else 1.0
            bonds.append((i, j))
            weights.append(w)
    total = sum(weights)
    weights = np.array(weights) / total
    return bonds, weights

def crossing_fraction(L, bonds, weights):
    cut = L // 2
    f = 0.0
    for (i, j), w in zip(bonds, weights):
        if i < cut and j >= cut:
            f += w
    return f

# ============================================================
# Trajectory
# ============================================================

def run_trajectory(L, alpha, p, depth, bonds, cum_w):
    """Run one trajectory, return final half-chain entropy."""
    psi = np.zeros(1 << L, dtype=complex)
    psi[0] = 1.0

    for step in range(depth):
        if np.random.random() < p:
            q = np.random.randint(L)
            _, psi = measure_z(psi, L, q)
        else:
            r = np.random.random()
            k = np.searchsorted(cum_w, r)
            k = min(k, len(bonds) - 1)
            q1, q2 = bonds[k]
            U = random_haar_2qubit()
            psi = apply_two_qubit_gate(psi, L, q1, q2, U)

    return half_chain_entropy(psi, L)

# ============================================================
# Main
# ============================================================

def main():
    np.random.seed(42)
    os.makedirs("results", exist_ok=True)

    # REVISION: added L=16; more α values; increased trajectories
    Ls = [8, 10, 12, 14, 16]
    alphas = [0.0, 0.5, 1.0, 1.5, 2.0]
    p_vals = [0.1, 0.3, 0.5, 0.7, 0.9]
    depth_factor = 20

    def n_traj(L):
        if L <= 10: return 200
        if L <= 12: return 100
        if L <= 14: return 50
        return 20   # L=16: 2^16 = 65536 dim, still feasible with vectorization

    print("=" * 60)
    print("  Haar-Random Gate Verification (Revised)")
    print("=" * 60)
    print(f"  Ls = {Ls}")
    print(f"  αs = {alphas}")
    print(f"  ps = {p_vals}")
    print()

    with open("results/haar_check.csv", "w") as f:
        f.write("alpha,p,L,S_mean,S_err,S_over_Smax,f_cross\n")

        t0 = time.time()
        for alpha in alphas:
            print(f"--- α = {alpha:.1f} ---")
            for L in Ls:
                bonds, weights = make_bonds(L, alpha)
                cum_w = np.cumsum(weights)
                fc = crossing_fraction(L, bonds, weights)
                depth = depth_factor * L
                nt = n_traj(L)

                for p in p_vals:
                    t_start = time.time()
                    Ss = []
                    for _ in range(nt):
                        S = run_trajectory(L, alpha, p, depth, bonds, cum_w)
                        Ss.append(S)

                    mean_S = np.mean(Ss)
                    err_S = np.std(Ss) / np.sqrt(nt)
                    Smax = L / 2.0
                    s_ratio = mean_S / Smax
                    dt = time.time() - t_start

                    f.write(f"{alpha:.2f},{p:.2f},{L},"
                            f"{mean_S:.4f},{err_S:.4f},{s_ratio:.4f},{fc:.4f}\n")

                    print(f"  L={L:3d} p={p:.1f} <S/Smax>={s_ratio:.3f}"
                          f" ({nt} traj, {dt:.1f}s)")

            print()

    elapsed = time.time() - t0
    print("=" * 60)
    print(f"  DONE. {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  → results/haar_check.csv")
    print("=" * 60)

    # Quick β check
    print("\n  Quick β check at p=0.5:")
    import csv
    data = {}
    with open("results/haar_check.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (float(row['alpha']), float(row['p']), int(row['L']))
            data[key] = float(row['S_mean'])

    for alpha in [0.0, 0.5, 1.0, 2.0]:
        S_vals, L_vals = [], []
        for L in Ls:
            key = (alpha, 0.5, L)
            if key in data and data[key] > 0.01:
                S_vals.append(data[key])
                L_vals.append(L)
        if len(S_vals) >= 3:
            log_L = np.log(L_vals)
            log_S = np.log(S_vals)
            beta = np.polyfit(log_L, log_S, 1)[0]
            label = 'volume-law' if beta > 0.7 else 'sub-volume'
            print(f"    α={alpha:.1f}: β ≈ {beta:.2f} ({label})")

if __name__ == "__main__":
    main()
