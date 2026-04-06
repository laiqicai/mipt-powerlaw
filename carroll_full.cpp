/*
 * ================================================================
 *  carroll_full.cpp — Unified simulation for PRL manuscript
 *  "Elimination of MIPTs in Circuits with Power-Law Connectivity"
 * ================================================================
 *
 *  Single code generating ALL data for the paper + supplement.
 *
 *  Compile:  g++ -O3 -std=c++17 -o carroll carroll_full.cpp
 *  Run:      ./carroll --all          (everything, ~hours)
 *            ./carroll --quick        (fast sanity check, ~2 min)
 *            ./carroll --standard     (Fig 1a,d: NN brickwork)
 *            ./carroll --longrange    (Fig 1b,c: α=0 long-range)
 *            ./carroll --alpha        (Fig 2, Table 1: α scan)
 *            ./carroll --deltaS       (Fig S1: rate eq. validation)
 *            ./carroll --fss          (Fig S2: finite-size scaling)
 *            ./carroll --marginal     (Fig S3: α=1 marginal case)
 *            ./carroll --therm        (Fig S4: thermalization)
 *            ./carroll --codedist     (Fig 3/S5: QEC code distance)
 *
 *  Output:   results/ (*.csv)
 * ================================================================
 */

#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <filesystem>
#include <iomanip>
#include <cassert>
#include <map>
#include <set>
#include <functional>

// ================================================================
//  §1  PAULI PHASE TABLE
// ================================================================
// Encoding: p = 2*x + z  →  I=0, Z=1, X=2, Y=3
//
//  NOTE: This differs from the Aaronson-Gottesman (2004) convention
//  where (x=1,z=1) represents XZ = -iY.  Here (x=1,z=1) represents
//  Y directly.  The G_TABLE and gate updates are self-consistent
//  with this choice.
//
// G_TABLE[p1][p2] = phase exponent when multiplying Pauli p1 * p2
//   i.e.  σ_{p1} σ_{p2} = i^{G[p1][p2]} σ_{p1⊕p2}
static const int8_t G_TABLE[4][4] = {
    {0,0,0,0},  // I·{I,Z,X,Y}
    {0,0,1,3},  // Z·{I,Z,X,Y}
    {0,3,0,1},  // X·{I,Z,X,Y}
    {0,1,3,0}   // Y·{I,Z,X,Y}
};

// ================================================================
//  §2  RNG (xoshiro256**)
// ================================================================
struct Rng {
    uint64_t s[4];
    void seed(uint64_t v) {
        for (int i = 0; i < 4; i++) {
            v += 0x9e3779b97f4a7c15ULL;
            uint64_t z = v;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            s[i] = z ^ (z >> 31);
        }
    }
    uint64_t next() {
        uint64_t r = ((s[1]*5) << 7 | (s[1]*5) >> 57) * 9;
        uint64_t t = s[1] << 17;
        s[2]^=s[0]; s[3]^=s[1]; s[1]^=s[2]; s[0]^=s[3];
        s[2]^=t; s[3]=(s[3]<<45)|(s[3]>>19);
        return r;
    }
    int    randint(int n) { return (int)(next() % (uint64_t)n); }
    double uniform()      { return (next()>>11) * (1.0/(1ULL<<53)); }
};
static Rng rng;

// ================================================================
//  §3  STABILIZER TABLEAU
// ================================================================
//
//  n generators, each described by (x[g][q], z[g][q], r[g])
//  where σ_g = i^{r[g]} ⊗_q  X^{x[g][q]} Z^{z[g][q]}
//
//  IMPORTANT: n = total qubits. For code distance mode, n = 2L.
//
struct CliffordState {
    int n;
    std::vector<uint8_t> x, z;   // x[g*n+q], z[g*n+q]
    std::vector<int8_t>  r;      // phase: i^{r[g]}

    CliffordState(int n_) : n(n_), x(n_*n_,0), z(n_*n_,0), r(n_,0) {
        // Initial state |0...0⟩: stabilized by +Z_j for j=0..n-1
        for (int i = 0; i < n; i++) z[i*n+i] = 1;
    }

    // --- accessors ---
    uint8_t& X(int g, int q)       { return x[g*n+q]; }
    uint8_t& Z(int g, int q)       { return z[g*n+q]; }
    const uint8_t& X(int g, int q) const { return x[g*n+q]; }
    const uint8_t& Z(int g, int q) const { return z[g*n+q]; }

    // --- Clifford gates ---
    void hadamard(int q) {
        for (int i = 0; i < n; i++) {
            r[i] = (r[i] + 2*X(i,q)*Z(i,q)) % 4;
            std::swap(X(i,q), Z(i,q));
        }
    }
    void phase_gate(int q) {
        for (int i = 0; i < n; i++) {
            r[i] = (r[i] + 2*X(i,q)*Z(i,q)) % 4;
            Z(i,q) ^= X(i,q);
        }
    }
    void cnot(int c, int t) {
        for (int i = 0; i < n; i++) {
            // Phase: XcZt(1⊕Xt⊕Zc) contributes sign
            int s = X(i,c)*Z(i,t)*(1 ^ X(i,t) ^ Z(i,c));
            r[i] = (r[i] + 2*s) % 4;
            X(i,t) ^= X(i,c);
            Z(i,c) ^= Z(i,t);
        }
    }

    // --- row multiplication: row_dst *= row_src ---
    void rowmult(int src, int dst) {
        int gt = 0;
        for (int j = 0; j < n; j++) {
            int ps = 2*X(src,j) + Z(src,j);
            int pd = 2*X(dst,j) + Z(dst,j);
            gt += G_TABLE[ps][pd];
        }
        r[dst] = ((int)r[src] + (int)r[dst] + gt%4 + 400) % 4;
        for (int j = 0; j < n; j++) {
            X(dst,j) ^= X(src,j);
            Z(dst,j) ^= Z(src,j);
        }
    }

    // --- random single-qubit Clifford ---
    void random_single(int q) {
        if (rng.randint(2)) hadamard(q);
        int k = rng.randint(4);
        for (int i = 0; i < k; i++) phase_gate(q);
        if (rng.randint(2)) hadamard(q);
    }

    // --- random two-qubit Clifford ---
    void random_two(int q1, int q2) {
        random_single(q1); random_single(q2);
        int c = rng.randint(3);
        if      (c == 0) cnot(q1, q2);
        else if (c == 1) cnot(q2, q1);
        else { hadamard(q2); cnot(q1, q2); hadamard(q2); }
        random_single(q1); random_single(q2);
    }

    // --- projective Z-measurement on qubit q ---
    //  Returns measurement outcome (0 or 1).
    //  Modifies tableau: if non-deterministic, collapses one generator.
    void measure_z(int q) {
        // Find generators anticommuting with Z_q (those with X(i,q)=1)
        int first = -1;
        for (int i = 0; i < n; i++) {
            if (X(i,q)) {
                if (first == -1) first = i;
                else rowmult(first, i);  // make i commute with Z_q
            }
        }
        if (first == -1) return;  // deterministic: no change
        // Non-deterministic: collapse generator 'first'
        int out = rng.randint(2);
        for (int j = 0; j < n; j++) { X(first,j) = 0; Z(first,j) = 0; }
        Z(first,q) = 1;
        r[first] = 2 * out;
    }

    // --- half-chain entanglement entropy ---
    //  S = rank(B-part of tableau) - |B|
    //  where B = qubits [cut, n)
    int entropy(int cut = -1) const {
        if (cut < 0) cut = n / 2;
        int nB = n - cut;
        // Extract B-part: for each generator, take (x,z) of qubits [cut,n)
        std::vector<uint8_t> B(n * 2*nB, 0);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < nB; j++) {
                B[i*2*nB + j]      = x[i*n + cut+j];
                B[i*2*nB + nB + j] = z[i*n + cut+j];
            }
        // Gaussian elimination for rank
        int rank = 0;
        for (int col = 0; col < 2*nB && rank < n; col++) {
            int piv = -1;
            for (int row = rank; row < n; row++)
                if (B[row*2*nB + col]) { piv = row; break; }
            if (piv == -1) continue;
            if (piv != rank)
                for (int k = 0; k < 2*nB; k++)
                    std::swap(B[rank*2*nB+k], B[piv*2*nB+k]);
            for (int row = 0; row < n; row++)
                if (row != rank && B[row*2*nB + col])
                    for (int k = 0; k < 2*nB; k++)
                        B[row*2*nB+k] ^= B[rank*2*nB+k];
            rank++;
        }
        return rank - nB;  // S = rank - |B|
    }
};

// ================================================================
//  §4  SELF-TESTS
// ================================================================
void self_tests() {
    std::cout << "  Self-tests..." << std::flush;
    // Product state S=0
    { CliffordState s(4); assert(s.entropy(2) == 0); }
    // Bell pair (H on q0, CNOT 0→1): S(cut=1)=1
    { CliffordState s(4); s.hadamard(0); s.cnot(0,1); assert(s.entropy(1)==1); }
    // Two cross-cut Bell pairs: S(cut=2)=2
    { CliffordState s(4);
      s.hadamard(0); s.cnot(0,2);
      s.hadamard(1); s.cnot(1,3);
      assert(s.entropy(2)==2); }
    // Measurement collapses entanglement
    { CliffordState s(4); s.hadamard(0); s.cnot(0,1);
      s.measure_z(0); assert(s.entropy(1)==0); }
    // S-gate: S|+> = |+i>, stabilized by +Y
    { CliffordState s(1); s.hadamard(0); s.phase_gate(0);
      assert(s.X(0,0)==1 && s.Z(0,0)==1 && s.r[0]==0); } // +Y
    // S²|+> = |->, stabilized by -X
    { CliffordState s(1); s.hadamard(0);
      s.phase_gate(0); s.phase_gate(0);
      assert(s.X(0,0)==1 && s.Z(0,0)==0 && s.r[0]==2); } // -X
    // S⁴ = I
    { CliffordState s(1); s.hadamard(0);
      for(int i=0;i<4;i++) s.phase_gate(0);
      assert(s.X(0,0)==1 && s.Z(0,0)==0 && s.r[0]==0); } // back to +X
    // S on |0>: Z-stabilizer unchanged
    { CliffordState s(1); s.phase_gate(0);
      assert(s.X(0,0)==0 && s.Z(0,0)==1 && s.r[0]==0); }
    // Random circuit builds entropy
    { rng.seed(456); CliffordState s(16);
      for (int i=0;i<100;i++){int q1=rng.randint(16),q2;
        do{q2=rng.randint(16);}while(q2==q1);
        s.random_two(q1,q2);}
      assert(s.entropy()>0); }
    std::cout << " ALL PASSED\n";
}

// ================================================================
//  §5  BOND STRUCTURES
// ================================================================
struct Bond { int i, j; };

// Nearest-neighbor (for brickwork)
void bonds_standard(int L, std::vector<Bond>& b, std::vector<double>& w) {
    b.clear(); w.clear();
    for (int i = 0; i < L-1; i++) { b.push_back({i,i+1}); w.push_back(1.0); }
    double s=0; for(auto v:w)s+=v; for(auto&v:w)v/=s;
}

// Power-law: w(d) = d^{-α}
void bonds_powerlaw(int L, double alpha, std::vector<Bond>& b, std::vector<double>& w) {
    b.clear(); w.clear();
    for (int i = 0; i < L; i++)
        for (int j = i+1; j < L; j++) {
            double wt = (alpha > 0.001) ? std::pow((double)(j-i), -alpha) : 1.0;
            b.push_back({i,j}); w.push_back(wt);
        }
    double s=0; for(auto v:w)s+=v; for(auto&v:w)v/=s;
}

int weighted_choice(const std::vector<double>& w) {
    double r = rng.uniform(), c = 0;
    for (int i = 0; i < (int)w.size(); i++) { c += w[i]; if (r < c) return i; }
    return (int)w.size()-1;
}

// Crossing fraction: probability a random bond crosses cut at L/2
double crossing_fraction(int L, const std::vector<Bond>& b, const std::vector<double>& w) {
    double f = 0;
    for (int i = 0; i < (int)b.size(); i++)
        if (b[i].i < L/2 && b[i].j >= L/2) f += w[i];
    return f;
}

// ================================================================
//  §6  TRAJECTORY FUNCTIONS
// ================================================================

// 6a. Brickwork (nearest-neighbor, parallel layers)
int run_brickwork(int L, double p, int n_layers) {
    CliffordState st(L);
    for (int layer = 0; layer < n_layers; layer++) {
        int off = layer % 2;
        for (int q = off; q+1 < L; q += 2)
            st.random_two(q, q+1);
        for (int q = 0; q < L; q++)
            if (rng.uniform() < p) st.measure_z(q);
    }
    return st.entropy();
}

// 6b. Random-bond (power-law, serial steps)
int run_random(int L, double p, int depth,
               const std::vector<Bond>& bonds, const std::vector<double>& weights) {
    CliffordState st(L);
    for (int s = 0; s < depth; s++) {
        if (rng.uniform() < p) st.measure_z(rng.randint(L));
        else { int k = weighted_choice(weights); st.random_two(bonds[k].i, bonds[k].j); }
    }
    return st.entropy();
}

// 6c. Random-bond with per-step ΔS tracking (for rate equation validation)
//  Bins: ΔS conditioned on (gate-crossing, gate-noncrossing, measurement)
//  as a function of current S/Smax, in steady state only.
struct DeltaSBin {
    double sum_gate_cross = 0;   int cnt_gate_cross = 0;
    double sum_gate_nc    = 0;   int cnt_gate_nc    = 0;
    double sum_meas       = 0;   int cnt_meas       = 0;
};

void run_deltaS(int L, double p, int depth, int n_traj, int n_bins,
                const std::vector<Bond>& bonds, const std::vector<double>& weights,
                std::vector<DeltaSBin>& bins) {
    bins.assign(n_bins, DeltaSBin{});
    int cut = L/2;
    double Smax = L/2.0;
    int warmup = depth/2;

    for (int t = 0; t < n_traj; t++) {
        CliffordState st(L);
        int S_cur = 0;
        for (int step = 0; step < depth; step++) {
            if (step < warmup) {
                // Just evolve, don't record
                if (rng.uniform()<p) st.measure_z(rng.randint(L));
                else { int k=weighted_choice(weights); st.random_two(bonds[k].i,bonds[k].j); }
                if (step == warmup-1) S_cur = st.entropy();
                continue;
            }
            double s_ratio = std::clamp((double)S_cur/Smax, 0.0, 1.0);
            int bi = std::min(n_bins-1, (int)(s_ratio * n_bins));

            if (rng.uniform() < p) {
                st.measure_z(rng.randint(L));
                int S_new = st.entropy();
                bins[bi].sum_meas += (S_new - S_cur);
                bins[bi].cnt_meas++;
                S_cur = S_new;
            } else {
                int k = weighted_choice(weights);
                bool crosses = (bonds[k].i < cut && bonds[k].j >= cut);
                st.random_two(bonds[k].i, bonds[k].j);
                int S_new = st.entropy();
                int dS = S_new - S_cur;
                if (crosses) { bins[bi].sum_gate_cross += dS; bins[bi].cnt_gate_cross++; }
                else         { bins[bi].sum_gate_nc    += dS; bins[bi].cnt_gate_nc++;    }
                S_cur = S_new;
            }
        }
    }
}

// ================================================================
//  §7  CODE DISTANCE VIA REFERENCE SYSTEM
// ================================================================
//
//  Algorithm (Gullans & Huse, PRX 2020):
//   1. Create 2L qubits: ref [0,L) + sys [L,2L)
//   2. Initialize L Bell pairs: CNOT(H(ref_i), sys_i)
//   3. Run circuit on system qubits only
//   4. k = entropy(cut=L) = #logical qubits
//   5. Gaussian elimination on ref qubits → separate normalizer (pivots)
//      from stabilizers (non-pivots)
//   6. Greedy weight-reduce pivot generators using stabilizers
//   7. d = min system-weight among non-trivially-supported pivots
//
//  Key subtlety: after GE on ref, pivots that reduce to weight-0 on
//  system are "dressed stabilizers" (classical correlations), NOT
//  logical operators.  Only weight>0 pivots contribute to d.

int sys_weight(const CliffordState& st, int g, int L) {
    int w = 0;
    for (int j = L; j < st.n; j++)
        if (st.X(g,j) || st.Z(g,j)) w++;
    return w;
}

int sys_weight_xor(const CliffordState& st, int a, int b, int L) {
    int w = 0;
    for (int j = L; j < st.n; j++) {
        if ((st.X(a,j)^st.X(b,j)) || (st.Z(a,j)^st.Z(b,j))) w++;
    }
    return w;
}

void greedy_reduce(CliffordState& st, int gen, const std::vector<int>& stabs, int L) {
    for (int pass = 0; pass < 100; pass++) {
        int cur = sys_weight(st, gen, L);
        if (cur <= 1) break;
        bool imp = false;
        // Single stabilizer
        for (int si : stabs) {
            if (sys_weight_xor(st, gen, si, L) < cur) {
                st.rowmult(si, gen);
                cur = sys_weight(st, gen, L);
                imp = true;
            }
        }
        // Pair of stabilizers
        if (!imp) {
            int ns = (int)stabs.size();
            for (int a = 0; a < ns && !imp; a++)
                for (int b = a+1; b < ns && !imp; b++) {
                    int w = 0;
                    for (int j = L; j < st.n; j++) {
                        uint8_t nx = st.X(gen,j)^st.X(stabs[a],j)^st.X(stabs[b],j);
                        uint8_t nz = st.Z(gen,j)^st.Z(stabs[a],j)^st.Z(stabs[b],j);
                        if (nx||nz) w++;
                    }
                    if (w < cur) {
                        st.rowmult(stabs[a], gen);
                        st.rowmult(stabs[b], gen);
                        cur = sys_weight(st, gen, L);
                        imp = true;
                    }
                }
        }
        if (!imp) break;
    }
}

struct CodeProps { int n, k, d; };

CodeProps compute_code(int L, double alpha, double p, int depth) {
    int N = 2*L;
    CliffordState st(N);
    // Bell pairs
    for (int i = 0; i < L; i++) { st.hadamard(i); st.cnot(i, L+i); }
    // System bonds
    std::vector<Bond> bonds; std::vector<double> weights;
    bonds_powerlaw(L, alpha, bonds, weights);
    // Circuit on system qubits [L, 2L)
    for (int step = 0; step < depth; step++) {
        if (rng.uniform() < p) st.measure_z(L + rng.randint(L));
        else { int b = weighted_choice(weights); st.random_two(L+bonds[b].i, L+bonds[b].j); }
    }
    int k = st.entropy(L);
    if (k == 0) return {L, 0, 0};

    // GE on reference qubits [0,L): identify pivots vs stabilizers
    std::vector<bool> used(N, false);
    std::vector<int> pivots, stabs;
    // Phase 1: X-pivots
    for (int rq = 0; rq < L; rq++) {
        int found = -1;
        for (int g = 0; g < N; g++) { if (!used[g] && st.X(g,rq)) { found=g; break; } }
        if (found == -1) continue;
        used[found] = true; pivots.push_back(found);
        for (int g = 0; g < N; g++)
            if (g != found && st.X(g,rq)) st.rowmult(found, g);
    }
    // Phase 2: Z-pivots
    for (int rq = 0; rq < L; rq++) {
        int found = -1;
        for (int g = 0; g < N; g++) { if (!used[g] && st.Z(g,rq)) { found=g; break; } }
        if (found == -1) continue;
        used[found] = true; pivots.push_back(found);
        for (int g = 0; g < N; g++)
            if (g != found && st.Z(g,rq)) st.rowmult(found, g);
    }
    for (int g = 0; g < N; g++) if (!used[g]) stabs.push_back(g);

    // Weight-reduce pivots and find d
    int min_d = L+1;
    std::vector<int> logicals;
    for (int pg : pivots) {
        greedy_reduce(st, pg, stabs, L);
        int w = sys_weight(st, pg, L);
        if (w > 0) { min_d = std::min(min_d, w); logicals.push_back(pg); }
    }
    // Try products of logical pairs
    for (int a = 0; a < (int)logicals.size(); a++)
        for (int b = a+1; b < (int)logicals.size(); b++) {
            int w = sys_weight_xor(st, logicals[a], logicals[b], L);
            if (w > 0) min_d = std::min(min_d, w);
        }
    if (min_d > L) min_d = 0;
    return {L, k, min_d};
}

// ================================================================
//  §8  HELPERS
// ================================================================

std::vector<double> linspace(double a, double b, int n) {
    std::vector<double> v(n);
    for (int i = 0; i < n; i++) v[i] = a + (b-a)*i/(n-1);
    return v;
}

int depth_std(int L) { return 4*L; }
int depth_lr(int L)  { return 20*L; }
int n_traj_for(int L) {
    if (L <= 64)  return 400;
    if (L <= 128) return 300;
    if (L <= 256) return 200;
    return 150;
}

struct DataPoint { double p, S_mean, S_err; };

// Compute S(p) curve for one L
std::vector<DataPoint> compute_curve(
    int L, const std::vector<double>& ps, int n_traj, int depth,
    const std::string& mode,
    const std::vector<Bond>& bonds, const std::vector<double>& weights,
    const std::string& label)
{
    std::vector<DataPoint> res;
    auto t0 = std::chrono::steady_clock::now();
    for (int ip = 0; ip < (int)ps.size(); ip++) {
        double p = ps[ip];
        double sS=0, sS2=0;
        for (int t = 0; t < n_traj; t++) {
            int S = (mode=="brickwork") ?
                run_brickwork(L, p, depth) :
                run_random(L, p, depth, bonds, weights);
            sS += S; sS2 += S*S;
        }
        double m = sS/n_traj;
        double v = sS2/n_traj - m*m;
        double e = std::sqrt(std::max(0.0,v)/n_traj);
        auto now = std::chrono::steady_clock::now();
        double el = std::chrono::duration<double>(now-t0).count();
        double eta = el/(ip+1)*((int)ps.size()-ip-1);
        std::cout << "  [" << label << " L=" << L << "] "
                  << (ip+1) << "/" << ps.size()
                  << " p=" << std::fixed << std::setprecision(3) << p
                  << " <S>=" << std::setprecision(2) << m
                  << " [" << (int)el << "s, ~" << (int)eta << "s]\n";
        res.push_back({p, m, e});
    }
    return res;
}

// Bootstrap β from raw trajectories
struct BetaResult { double beta, err; };
BetaResult bootstrap_beta(const std::vector<int>& Ls,
                           const std::vector<std::vector<int>>& raw,
                           int n_boot=200) {
    int nL = (int)Ls.size();
    auto fit = [&](const std::vector<double>& means) {
        double sx=0,sy=0,sxx=0,sxy=0;
        for (int i=0;i<nL;i++){
            double lx=std::log((double)Ls[i]);
            double ly=std::log(std::max(0.01,means[i]));
            sx+=lx;sy+=ly;sxx+=lx*lx;sxy+=lx*ly;
        }
        return (nL*sxy-sx*sy)/(nL*sxx-sx*sx);
    };
    // Point estimate
    std::vector<double> means(nL);
    for (int i=0;i<nL;i++){
        double s=0; for(auto v:raw[i])s+=v; means[i]=s/raw[i].size();
    }
    double beta_pt = fit(means);
    // Bootstrap
    std::vector<double> betas(n_boot);
    for (int b=0;b<n_boot;b++){
        std::vector<double> bm(nL);
        for (int i=0;i<nL;i++){
            int nt=(int)raw[i].size(); double s=0;
            for(int j=0;j<nt;j++) s+=raw[i][rng.randint(nt)];
            bm[i]=s/nt;
        }
        betas[b]=fit(bm);
    }
    double mb=0; for(auto v:betas)mb+=v; mb/=n_boot;
    double vb=0; for(auto v:betas)vb+=(v-mb)*(v-mb); vb/=(n_boot-1);
    return {beta_pt, std::sqrt(vb)};
}

// ================================================================
//  §9  MODE IMPLEMENTATIONS
// ================================================================

// 9a. Standard brickwork NN
void mode_standard(bool quick) {
    std::cout << "\n==== MODE: STANDARD (NN brickwork) ====\n";
    std::vector<int> Ls = quick ? std::vector<int>{16,32,64}
                                : std::vector<int>{16,32,64,128,256};
    auto ps = linspace(0.01, 0.50, quick?15:25);
    std::ofstream csv("results/standard.csv");
    csv << "p";
    for (int L:Ls) csv << ",S_mean_L" << L << ",S_err_L" << L;
    csv << "\n";
    std::vector<std::vector<DataPoint>> all;
    for (int L : Ls) {
        std::vector<Bond> b; std::vector<double> w;
        bonds_standard(L, b, w);
        all.push_back(compute_curve(L, ps, n_traj_for(L), depth_std(L),
                                    "brickwork", b, w, "std"));
    }
    for (int ip=0;ip<(int)ps.size();ip++){
        csv << std::setprecision(6) << ps[ip];
        for (int il=0;il<(int)Ls.size();il++)
            csv << "," << all[il][ip].S_mean << "," << all[il][ip].S_err;
        csv << "\n";
    }
    csv.close();
    std::cout << "  → results/standard.csv\n";
}

// 9b. Long-range α=0
void mode_longrange(bool quick) {
    std::cout << "\n==== MODE: LONG-RANGE (α=0) ====\n";
    std::vector<int> Ls = quick ? std::vector<int>{16,32,64}
                                : std::vector<int>{16,32,64,128,256,512};
    auto ps = linspace(0.05, 0.95, quick?15:25);
    std::ofstream csv("results/carroll.csv");
    csv << "p";
    for (int L:Ls) csv << ",S_mean_L" << L << ",S_err_L" << L;
    csv << "\n";
    std::vector<std::vector<DataPoint>> all;
    for (int L : Ls) {
        std::vector<Bond> b; std::vector<double> w;
        bonds_powerlaw(L, 0.0, b, w);
        all.push_back(compute_curve(L, ps, n_traj_for(L), depth_lr(L),
                                    "random", b, w, "lr"));
    }
    for (int ip=0;ip<(int)ps.size();ip++){
        csv << std::setprecision(6) << ps[ip];
        for (int il=0;il<(int)Ls.size();il++)
            csv << "," << all[il][ip].S_mean << "," << all[il][ip].S_err;
        csv << "\n";
    }
    csv.close();
    std::cout << "  → results/carroll.csv\n";
}

// 9c. Alpha scan + bootstrap β
//  REVISION: L=256 added; trajectory count increased for α∈[0.7,1.0]
void mode_alpha(bool quick) {
    std::cout << "\n==== MODE: ALPHA SCAN (revised) ====\n";
    std::vector<double> alphas = {0.0,0.3,0.5,0.7,0.9,1.0,1.2,1.5,2.0};
    std::vector<int> Ls = quick ? std::vector<int>{32,64,128}
                                : std::vector<int>{32,64,128,256};
    std::vector<double> ps = {0.1,0.3,0.5,0.7,0.9};
    int nt_base = quick ? 80 : 200;

    std::ofstream csv("results/alpha_scan.csv");
    csv << "alpha,p,L,S_mean,S_err,S_over_Smax,f_cross,theory\n";
    std::ofstream bcsv("results/beta_table.csv");
    bcsv << "alpha,p,beta,beta_err\n";

    for (double alpha : alphas) {
        std::cout << "--- α=" << std::fixed << std::setprecision(1) << alpha << " ---\n";
        std::vector<std::vector<Bond>> aB(Ls.size());
        std::vector<std::vector<double>> aW(Ls.size());
        std::vector<double> aF(Ls.size());
        for (int il=0;il<(int)Ls.size();il++){
            bonds_powerlaw(Ls[il],alpha,aB[il],aW[il]);
            aF[il]=crossing_fraction(Ls[il],aB[il],aW[il]);
        }
        for (double p : ps) {
            // REVISION: adaptive trajectory count for critical region
            int nt = nt_base;
            if (!quick && alpha >= 0.7 && alpha <= 1.0) nt = 400;
            // L=256 is expensive: scale down but keep minimum
            // (will be applied per-L below)

            std::vector<std::vector<int>> raw(Ls.size());
            std::vector<double> means(Ls.size());
            for (int il=0;il<(int)Ls.size();il++){
                int L=Ls[il], depth=depth_lr(L);
                int nt_L = (L <= 128) ? nt : std::max(nt*2/3, 100);
                raw[il].resize(nt_L);
                double sS=0,sS2=0;
                for (int t=0;t<nt_L;t++){
                    int S=run_random(L,p,depth,aB[il],aW[il]);
                    raw[il][t]=S; sS+=S; sS2+=S*S;
                }
                double m=sS/nt_L, v=sS2/nt_L-m*m, e=std::sqrt(std::max(0.0,v)/nt_L);
                double Sm=L/2.0, th=(1-p)*aF[il]/((1-p)*aF[il]+p);
                means[il]=m;
                csv << std::setprecision(2)<<alpha<<","<<p<<","<<L
                    << std::setprecision(4)<<","<<m<<","<<e<<","<<m/Sm
                    <<","<<aF[il]<<","<<th<<"\n";
            }
            BetaResult br = bootstrap_beta(Ls, raw);
            bcsv << std::setprecision(2)<<alpha<<","<<p
                 << std::setprecision(3)<<","<<br.beta<<","<<br.err<<"\n";
            std::cout << "  p="<<std::setprecision(1)<<p
                      <<" β="<<std::setprecision(2)<<br.beta
                      <<"±"<<br.err<<"\n";
        }
    }
    csv.close(); bcsv.close();
    std::cout << "  → results/alpha_scan.csv, results/beta_table.csv\n";
}

// 9d. Per-step ΔS validation
void mode_deltaS() {
    std::cout << "\n==== MODE: ΔS VALIDATION ====\n";
    int L=64, nbins=10, depth=20*L, nt=100;
    std::vector<std::pair<double,double>> configs =
        {{0.0,0.1},{0.0,0.3},{0.0,0.5},{0.0,0.7},{0.5,0.3},{0.5,0.5}};

    std::ofstream csv("results/delta_s_validation.csv");
    csv << "alpha,L,p,s_center,ds_gate_cross,cnt_gc,ds_gate_nc,cnt_gnc,ds_meas,cnt_m\n";

    for (auto [alpha,p] : configs) {
        std::cout << "  α="<<alpha<<" p="<<p<<"..."; std::cout.flush();
        std::vector<Bond> b; std::vector<double> w;
        bonds_powerlaw(L, alpha, b, w);
        std::vector<DeltaSBin> bins;
        run_deltaS(L, p, depth, nt, nbins, b, w, bins);
        for (int i=0;i<nbins;i++){
            double sc = (i+0.5)/nbins;
            double dg = bins[i].cnt_gate_cross>0 ? bins[i].sum_gate_cross/bins[i].cnt_gate_cross : 0;
            double dn = bins[i].cnt_gate_nc>0    ? bins[i].sum_gate_nc/bins[i].cnt_gate_nc : 0;
            double dm = bins[i].cnt_meas>0        ? bins[i].sum_meas/bins[i].cnt_meas : 0;
            csv << std::setprecision(2)<<alpha<<","<<L<<","<<p
                <<std::setprecision(3)<<","<<sc
                <<std::setprecision(6)<<","<<dg<<","<<bins[i].cnt_gate_cross
                <<","<<dn<<","<<bins[i].cnt_gate_nc
                <<","<<dm<<","<<bins[i].cnt_meas<<"\n";
        }
        std::cout << " done\n";
    }
    csv.close();
    std::cout << "  → results/delta_s_validation.csv\n";
}

// 9e. Finite-size scaling near α_c
//  REVISION: extended to L=512; added α=0.5 reference; increased trajectories
void mode_fss(bool quick) {
    std::cout << "\n==== MODE: FINITE-SIZE SCALING (revised) ====\n";
    std::vector<double> alphas = {0.5,0.7,0.8,0.9,1.0,1.1,1.2};
    std::vector<int> Ls = quick ? std::vector<int>{32,64,128}
                                : std::vector<int>{32,64,128,256,512};
    std::vector<double> ps = {0.3,0.5,0.7,0.9};
    int nt_base = quick ? 80 : 200;

    std::ofstream csv("results/fss_near_critical.csv");
    csv << "alpha,p,L,S_mean,S_err,S_over_Smax\n";

    for (double alpha : alphas) {
        std::cout << "  α="<<std::setprecision(1)<<alpha<<"\n";
        for (double p : ps)
            for (int L : Ls) {
                int nt = (L<=128) ? nt_base : (L<=256) ? nt_base*2/3 : nt_base/3;
                nt = std::max(nt, 60);  // minimum 60 trajectories
                std::vector<Bond> b; std::vector<double> w;
                bonds_powerlaw(L, alpha, b, w);
                double sS=0,sS2=0;
                for (int t=0;t<nt;t++){
                    int S=run_random(L,p,depth_lr(L),b,w); sS+=S;sS2+=S*S;
                }
                double m=sS/nt, v=sS2/nt-m*m, e=std::sqrt(std::max(0.0,v)/nt);
                csv << std::setprecision(2)<<alpha<<","<<p<<","<<L
                    <<std::setprecision(4)<<","<<m<<","<<e<<","<<m/(L/2.0)<<"\n";
                std::cout << "    α="<<alpha<<" p="<<p<<" L="<<L
                          <<" S/Smax="<<std::setprecision(3)<<m/(L/2.0)
                          <<" ("<<nt<<" traj)\n";
            }
    }
    csv.close();
    std::cout << "  → results/fss_near_critical.csv\n";
}

// 9f. Marginal case α=1
//  REVISION: extended to L=512; increased trajectory counts
void mode_marginal(bool quick) {
    std::cout << "\n==== MODE: MARGINAL α=1 (revised) ====\n";
    double alpha = 1.0;
    std::vector<int> Ls = quick ? std::vector<int>{32,64,128}
                                : std::vector<int>{32,64,128,256,512};
    std::vector<double> ps = {0.1,0.3,0.5,0.7,0.9};

    std::ofstream csv("results/marginal_alpha1.csv");
    csv << "L,p,S_mean,S_err,S_over_Smax,f_cross,S_lnL_over_L\n";

    for (int L : Ls) {
        std::vector<Bond> b; std::vector<double> w;
        bonds_powerlaw(L, alpha, b, w);
        double f = crossing_fraction(L, b, w);
        int nt = (L<=128)?200:(L<=256?150:80);
        for (double p : ps) {
            double sS=0,sS2=0;
            for (int t=0;t<nt;t++){
                int S=run_random(L,p,depth_lr(L),b,w);sS+=S;sS2+=S*S;
            }
            double m=sS/nt,v=sS2/nt-m*m,e=std::sqrt(std::max(0.0,v)/nt);
            csv << L<<","<<std::setprecision(2)<<p
                <<std::setprecision(4)<<","<<m<<","<<e<<","<<m/(L/2.0)
                <<","<<f<<","<<m*std::log((double)L)/L<<"\n";
            std::cout << "  L="<<L<<" p="<<p
                      <<" S·lnL/L="<<std::setprecision(3)<<m*std::log((double)L)/L<<"\n";
        }
    }
    csv.close();
    std::cout << "  → results/marginal_alpha1.csv\n";
}

// 9g. Thermalization convergence
void mode_therm() {
    std::cout << "\n==== MODE: THERMALIZATION ====\n";
    struct Cfg { double alpha; int L; double p; };
    std::vector<Cfg> cfgs = {
        {0.0,128,0.3},{0.0,128,0.7},{0.5,128,0.5},
        {1.0,128,0.5},{1.5,128,0.5},{0.0,256,0.5}
    };
    int n_checks=10, nt=80;

    std::ofstream csv("results/thermalization_check.csv");
    csv << "alpha,L,p,depth,S_mean,S_err\n";

    for (auto& c : cfgs) {
        std::vector<Bond> b; std::vector<double> w;
        bonds_powerlaw(c.L, c.alpha, b, w);
        int max_d = depth_lr(c.L);
        for (int ci=1; ci<=n_checks; ci++) {
            int depth = max_d*ci/n_checks;
            double sS=0,sS2=0;
            for (int t=0;t<nt;t++){
                int S=run_random(c.L,c.p,depth,b,w);sS+=S;sS2+=S*S;
            }
            double m=sS/nt,v=sS2/nt-m*m,e=std::sqrt(std::max(0.0,v)/nt);
            csv << std::setprecision(2)<<c.alpha<<","<<c.L<<","<<c.p
                <<","<<depth<<std::setprecision(4)<<","<<m<<","<<e<<"\n";
        }
        std::cout << "  α="<<c.alpha<<" L="<<c.L<<" p="<<c.p<<" done\n";
    }
    csv.close();
    std::cout << "  → results/thermalization_check.csv\n";
}

// 9h. Code distance
void mode_codedist(bool quick) {
    std::cout << "\n==== MODE: CODE DISTANCE ====\n";
    std::vector<int> Ls = quick ? std::vector<int>{8,12,16,20}
                                : std::vector<int>{8,12,16,20,24,32};
    std::vector<double> alphas = {0.0,0.5,1.0,1.5,2.0};
    std::vector<double> ps = {0.05,0.1,0.2,0.3,0.5};
    auto ntf = [](int L)->int{
        if (L <= 12) return 200;
        if (L <= 20) return 100;
        if (L <= 32) return 50;
        return 20;
    };

    std::ofstream csv("results/code_distance.csv");
    csv << "alpha,p,L,k_mean,k_err,d_mean,d_err,rate,rel_d,n_valid\n";

    for (double alpha : alphas) {
        std::cout << "=== α="<<std::fixed<<std::setprecision(1)<<alpha<<" ===\n";
        for (int L : Ls) {
            int nt=ntf(L), depth=depth_lr(L);
            for (double p : ps) {
                double sk=0,sk2=0,sd=0,sd2=0; int nv=0;
                for (int t=0;t<nt;t++){
                    CodeProps cp=compute_code(L,alpha,p,depth);
                    sk+=cp.k;sk2+=cp.k*cp.k;
                    if(cp.k>0&&cp.d>0){sd+=cp.d;sd2+=cp.d*cp.d;nv++;}
                }
                double km=sk/nt,ke=std::sqrt(std::max(0.0,sk2/nt-km*km)/nt);
                double dm=(nv>0)?sd/nv:0,de=(nv>1)?std::sqrt(std::max(0.0,sd2/nv-dm*dm)/nv):0;
                csv << std::setprecision(2)<<alpha<<","<<p<<","<<L
                    <<","<<km<<","<<ke<<","<<dm<<","<<de
                    <<std::setprecision(4)<<","<<km/L<<","<<dm/L<<","<<nv<<"\n";
                std::cout << "  L="<<std::setw(3)<<L<<" p="<<std::setprecision(2)<<p
                          <<"  [["<<L<<","<<std::setprecision(1)<<km<<","<<dm<<"]]"
                          <<"  d/n="<<std::setprecision(3)<<dm/L
                          <<"  ("<<nv<<"/"<<nt<<")\n";
            }
        }
    }
    csv.close();

    // Fit d ~ L^gamma summary
    std::cout << "\n--- d vs L scaling ---\n";
    std::ifstream fin("results/code_distance.csv");
    std::string hdr; std::getline(fin,hdr);
    std::map<std::pair<double,double>,std::vector<std::pair<int,double>>> data;
    std::string line;
    while(std::getline(fin,line)){
        double a,p2,km,ke,dm,de,rt,rd; int L2,nv2;
        if(sscanf(line.c_str(),"%lf,%lf,%d,%lf,%lf,%lf,%lf,%lf,%lf,%d",
                  &a,&p2,&L2,&km,&ke,&dm,&de,&rt,&rd,&nv2)>=8)
            if(dm>0.1) data[{a,p2}].push_back({L2,dm});
    }
    for (auto& [key,pts]:data) {
        if(pts.size()<3)continue;
        double sx=0,sy=0,sxx=0,sxy=0;int nn=0;
        for(auto&[L2,dm]:pts){double lx=std::log(L2),ly=std::log(dm);
            sx+=lx;sy+=ly;sxx+=lx*lx;sxy+=lx*ly;nn++;}
        double gamma=(nn*sxy-sx*sy)/(nn*sxx-sx*sx);
        std::cout << "  α="<<std::setprecision(1)<<key.first
                  <<" p="<<std::setprecision(2)<<key.second
                  <<" : d ~ L^"<<std::setprecision(2)<<gamma
                  <<(gamma>0.3?" ← GOOD CODE":"")
                  <<"  ("<<nn<<" pts)\n";
    }
    std::cout << "  → results/code_distance.csv\n";
}

// ================================================================
//  §9i  GATE ENSEMBLE SENSITIVITY CHECK (REVISION — Concern 6)
// ================================================================
//
//  Compare three different 2-qubit Clifford gate distributions at
//  L=64, α=0 to verify that S(p) is insensitive to gate choice.

// Gate variant 2: CNOT-only (no CZ)
void random_two_v2(CliffordState& st, int q1, int q2) {
    st.random_single(q1); st.random_single(q2);
    st.cnot(q1, q2);
    st.random_single(q1); st.random_single(q2);
}

// Gate variant 3: CNOT + SWAP (4-way choice)
void random_two_v3(CliffordState& st, int q1, int q2) {
    st.random_single(q1); st.random_single(q2);
    int c = rng.randint(4);
    if      (c == 0) st.cnot(q1, q2);
    else if (c == 1) st.cnot(q2, q1);
    else if (c == 2) { st.hadamard(q2); st.cnot(q1, q2); st.hadamard(q2); }
    else { st.cnot(q1,q2); st.cnot(q2,q1); st.cnot(q1,q2); } // SWAP
    st.random_single(q1); st.random_single(q2);
}

// Trajectory with selectable gate variant
int run_random_gv(int L, double p, int depth,
                  const std::vector<Bond>& bonds, const std::vector<double>& weights,
                  int gate_variant) {
    CliffordState st(L);
    for (int s = 0; s < depth; s++) {
        if (rng.uniform() < p) st.measure_z(rng.randint(L));
        else {
            int k = weighted_choice(weights);
            int q1 = bonds[k].i, q2 = bonds[k].j;
            switch (gate_variant) {
                case 1: st.random_two(q1, q2); break;
                case 2: random_two_v2(st, q1, q2); break;
                case 3: random_two_v3(st, q1, q2); break;
            }
        }
    }
    return st.entropy();
}

void mode_gate_ensemble() {
    std::cout << "\n==== MODE: GATE ENSEMBLE CHECK ====\n";
    int L = 64;
    double alpha = 0.0;
    int nt = 200;
    auto ps = linspace(0.05, 0.95, 15);
    std::vector<Bond> b; std::vector<double> w;
    bonds_powerlaw(L, alpha, b, w);
    int depth = depth_lr(L);

    std::ofstream csv("results/gate_ensemble_check.csv");
    csv << "gate_variant,p,S_mean,S_err,S_over_Smax\n";
    std::vector<std::string> names = {"", "original(CNOT+CZ)", "CNOT-only", "CNOT+SWAP"};

    for (int gv = 1; gv <= 3; gv++) {
        std::cout << "  Gate variant " << gv << ": " << names[gv] << "\n";
        for (double p : ps) {
            double sS=0, sS2=0;
            for (int t=0; t<nt; t++) {
                int S = run_random_gv(L, p, depth, b, w, gv);
                sS += S; sS2 += S*S;
            }
            double m = sS/nt, v = sS2/nt - m*m;
            double e = std::sqrt(std::max(0.0,v)/nt);
            csv << gv << "," << std::setprecision(4) << p
                << "," << m << "," << e << "," << m/(L/2.0) << "\n";
        }
    }
    csv.close();
    std::cout << "  → results/gate_ensemble_check.csv\n";
}

// ================================================================
//  §10  MAIN
// ================================================================
int main(int argc, char* argv[]) {
    std::set<std::string> flags;
    for (int i=1;i<argc;i++) flags.insert(argv[i]);

    bool quick = flags.count("--quick");
    bool all   = flags.count("--all");

    std::filesystem::create_directories("results");
    rng.seed(42);

    std::cout << "============================================================\n";
    std::cout << "  Carroll MIPT — Full Simulation Suite\n";
    std::cout << "============================================================\n";
    self_tests();

    auto t0 = std::chrono::steady_clock::now();

    if (all || flags.count("--standard"))  mode_standard(quick);
    if (all || flags.count("--longrange")) mode_longrange(quick);
    if (all || flags.count("--alpha"))     mode_alpha(quick);
    if (all || flags.count("--deltaS"))    mode_deltaS();
    if (all || flags.count("--fss"))       mode_fss(quick);
    if (all || flags.count("--marginal"))  mode_marginal(quick);
    if (all || flags.count("--therm"))     mode_therm();
    if (all || flags.count("--codedist"))  mode_codedist(quick);
    if (all || flags.count("--gates"))    mode_gate_ensemble();

    if (flags.empty()) {
        std::cout << "\nUsage: ./carroll [--all|--quick] [--standard] [--longrange]\n"
                  << "       [--alpha] [--deltaS] [--fss] [--marginal] [--therm]\n"
                  << "       [--codedist] [--gates]\n";
    }

    auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now()-t0).count();
    std::cout << "\n============================================================\n";
    std::cout << "  Total: " << (int)elapsed << "s ("
              << std::setprecision(1) << elapsed/60 << " min)\n";
    std::cout << "============================================================\n";
}
