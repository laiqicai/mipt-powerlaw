#!/usr/bin/env python3
"""
================================================================
  make_all_figures.py — Generate ALL publication figures
  
  Generates every figure for main text + supplement from CSV data.
  
  Required CSV files in DATADIR (default: results/):
    standard.csv              → Fig 1(a,d)
    carroll.csv               → Fig 1(b,c)
    alpha_scan.csv            → Fig 2, phase diagram, Table I
    beta_table.csv            → Table I (console)
    delta_s_validation.csv    → Fig S1
    fss_near_critical.csv     → Fig S2, S3, β(L) main
    marginal_alpha1.csv       → Fig S4
    haar_check.csv            → Fig S5
    thermalization_check.csv  → Fig S6
    free_fermion_mipt_gpu.csv → Fig S7 (Hamiltonian comparison)
    gate_ensemble_check.csv   → Fig S8

  Usage:
    python make_all_figures.py                    # all figures
    python make_all_figures.py --main             # main text only
    python make_all_figures.py --supplement       # supplement only
    DATADIR=my_results python make_all_figures.py # custom data dir
================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import csv, os, sys

rcParams.update({
    'font.size': 9, 'axes.labelsize': 10, 'legend.fontsize': 7,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.family': 'serif', 'mathtext.fontset': 'cm',
    'axes.linewidth': 0.6, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
})

DATADIR = os.environ.get('DATADIR', 'results')

COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
          '#a65628', '#f781bf', '#999999', '#66c2a5']
C_SIZE  = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02']
C_ALPHA = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#984ea3', '#a65628']

def load(name):
    path = f'{DATADIR}/{name}'
    if not os.path.exists(path):
        print(f"  [SKIP] {name} not found in {DATADIR}/")
        return None
    with open(path) as f:
        return list(csv.DictReader(f))

# ================================================================
# MAIN TEXT FIGURES
# ================================================================

def make_fig1():
    """Fig 1: NN vs α=0 — S/Smax, S vs L, susceptibility."""
    rows_std = load('standard.csv')
    rows_lr  = load('carroll.csv')
    if not rows_std or not rows_lr:
        print('  [SKIP] fig1 — missing data'); return

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.5))

    # (a) Standard NN
    ax = axes[0,0]
    cols = [c for c in rows_std[0] if c.startswith('S_mean_L')]
    Ls = sorted(int(c.replace('S_mean_L','')) for c in cols)
    for ci, L in enumerate(Ls):
        ps = [float(r['p']) for r in rows_std]
        Ss = [float(r[f'S_mean_L{L}'])/(L/2) for r in rows_std]
        ax.plot(ps, Ss, '-', color=C_SIZE[ci], lw=1.2, label=f'$L={L}$')
    ax.axvline(0.13, color='gray', ls=':', lw=0.8)
    ax.text(0.15, 0.85, r'$p_c\approx 0.13$', fontsize=8, color='gray',
            transform=ax.transAxes)
    ax.set_xlabel(r'$p$'); ax.set_ylabel(r'$S/S_{\max}$')
    ax.set_title(r'(a) Standard (nearest-neighbor)', fontsize=9)
    ax.legend(loc='upper right', ncol=2, framealpha=0.9)
    ax.set_xlim(0, 0.5); ax.set_ylim(0, 1.05)

    # (b) Long-range α=0
    ax = axes[0,1]
    cols = [c for c in rows_lr[0] if c.startswith('S_mean_L')]
    Ls = sorted(int(c.replace('S_mean_L','')) for c in cols)
    for ci, L in enumerate(Ls):
        ps = [float(r['p']) for r in rows_lr]
        Ss = [float(r[f'S_mean_L{L}'])/(L/2) for r in rows_lr]
        ax.plot(ps, Ss, '-', color=C_SIZE[ci], lw=1.2, label=f'$L={L}$')
    pt = np.linspace(0.01, 0.99, 200)
    st = (1-pt)*0.5 / ((1-pt)*0.5 + pt)
    ax.plot(pt, st, 'k--', lw=1, label=r'$(1{-}p)/(1{+}p)$')
    ax.set_xlabel(r'$p$'); ax.set_ylabel(r'$S/S_{\max}$')
    ax.set_title(r'(b) Power-law $\alpha = 0$ (uniform long-range)', fontsize=9)
    ax.text(0.5, 0.55, 'No $p_c < 1$', fontsize=9, color='#d95f02',
            ha='center', transform=ax.transAxes)
    ax.legend(loc='upper right', ncol=2, framealpha=0.9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)

    # (c) S vs L log-log
    ax = axes[1,0]
    for ci, ptarg in enumerate([0.31, 0.50, 0.69, 0.91, 0.95]):
        best = min(rows_lr, key=lambda r: abs(float(r['p']) - ptarg))
        p_actual = float(best['p'])
        Ss = [float(best[f'S_mean_L{L}']) for L in Ls]
        ax.plot(Ls, Ss, 'o-', color=C_SIZE[ci], ms=3, lw=1,
                label=f'$p={p_actual:.2f}$')
    ax.set_xscale('log', base=2); ax.set_yscale('log', base=2)
    ax.set_xlabel(r'$L$'); ax.set_ylabel(r'$S$')
    ax.set_title(r'(c) $S$ vs $L$: volume law at all $p$', fontsize=9)
    ax.legend(fontsize=6, loc='lower right')

    # (d) Susceptibility
    ax = axes[1,1]
    for label, rows_d, color, marker in [
        ('Standard (NN)', rows_std, '#377eb8', 's'),
        (r'$\alpha=0$ (long-range)', rows_lr, '#e41a1c', 'o')]:
        cols = [c for c in rows_d[0] if c.startswith('S_mean_L')]
        all_Ls = sorted(int(c.replace('S_mean_L','')) for c in cols)
        peaks = []
        for L in all_Ls:
            ps = [float(r['p']) for r in rows_d]
            Ss = [float(r[f'S_mean_L{L}']) for r in rows_d]
            dsdp = [-(Ss[i+1]-Ss[i])/(ps[i+1]-ps[i])/(L/2) for i in range(len(ps)-1)]
            peaks.append((L, max(dsdp)))
        ax.plot([x[0] for x in peaks], [x[1] for x in peaks],
                f'{marker}-', color=color, ms=4, lw=1, label=label)
    ax.set_xscale('log', base=2); ax.set_yscale('log', base=2)
    ax.set_xlabel(r'$L$'); ax.set_ylabel(r'$\max(-dS/dp)$')
    ax.set_title('(d) Susceptibility peak scaling', fontsize=9)
    ax.legend(fontsize=7)

    plt.tight_layout(pad=0.5)
    plt.savefig('fig1.pdf'); plt.close()
    print('  → fig1.pdf')


def make_fig2():
    """Fig 2: crossing fraction, entropy vs p, phase diagram, theory check."""
    rows = load('alpha_scan.csv')
    if not rows:
        print('  [SKIP] fig2 — missing data'); return

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.5))

    # (a) Crossing fraction f vs α
    ax = axes[0,0]
    by_L = {}
    for r in rows:
        L = int(r['L'])
        by_L.setdefault(L, {})[float(r['alpha'])] = float(r['f_cross'])
    for ci, L in enumerate([32, 64, 128]):
        if L not in by_L: continue
        alphas = sorted(by_L[L])
        ax.plot(alphas, [by_L[L][a] for a in alphas], 'o', color=C_SIZE[ci],
                ms=4, label=f'$L={L}$')
    a_th = np.linspace(0, 0.99, 100)
    ax.plot(a_th, 1 - 2**(a_th - 1), 'k--', lw=1.2, label=r'$1-2^{\alpha-1}$')
    ax.axvline(1.0, color='gray', ls=':', lw=0.7)
    ax.text(1.05, 0.35, r'$\alpha_c=1$', fontsize=8, color='gray')
    ax.set_xlabel(r'$\alpha$'); ax.set_ylabel(r'Crossing fraction $f$')
    ax.set_title('(a) Gate-crossing probability', fontsize=9)
    ax.legend(fontsize=7)

    # (b) Entropy vs p at L=128
    ax = axes[0,1]
    by_a = {}
    for r in rows:
        if int(r['L']) != 128: continue
        a = float(r['alpha'])
        by_a.setdefault(a, {'p':[], 's':[], 'th':[]})
        by_a[a]['p'].append(float(r['p']))
        by_a[a]['s'].append(float(r['S_over_Smax']))
        by_a[a]['th'].append(float(r['theory']))
    for ci, a in enumerate([0.0, 0.5, 1.0, 1.5, 2.0]):
        if a not in by_a: continue
        d = by_a[a]
        ax.plot(d['p'], d['s'], 'o', color=C_ALPHA[ci], ms=3, label=rf'$\alpha={a:.1f}$')
        ax.plot(d['p'], d['th'], '--', color=C_ALPHA[ci], lw=0.8, alpha=0.6)
    ax.set_xlabel(r'$p$'); ax.set_ylabel(r'$S/S_{\max}$')
    ax.set_title(r'(b) Entropy vs $p$ ($L=128$)', fontsize=9)
    ax.legend(fontsize=7)

    # (c) Phase diagram
    ax = axes[1,0]
    A = sorted(set(float(r['alpha']) for r in rows))
    P = sorted(set(float(r['p']) for r in rows))
    grid = {(float(r['alpha']), float(r['p'])): float(r['S_over_Smax'])
            for r in rows if int(r['L']) == 128}
    Z = np.array([[grid.get((a,p), 0) for p in P] for a in A])
    im = ax.pcolormesh(P, A, Z, cmap='RdYlBu_r', shading='nearest', vmin=0, vmax=0.9)
    ax.axhline(1.0, color='white', ls='--', lw=1.5)
    ax.text(0.5, 1.6, 'Area law\n(MIPT exists)', fontsize=7, ha='center',
            color='white', weight='bold')
    ax.text(0.5, 0.4, 'Volume law\n(no MIPT)', fontsize=7, ha='center',
            color='white', weight='bold')
    ax.set_xlabel('Measurement rate $p$')
    ax.set_ylabel(r'Power-law exponent $\alpha$')
    ax.set_title(r'(c) Phase diagram ($L=128$)', fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.85).set_label(r'$S/S_{\max}$', fontsize=8)

    # (d) Theory vs simulation
    ax = axes[1,1]
    for r in rows:
        if int(r['L']) != 128: continue
        ax.plot(float(r['theory']), float(r['S_over_Smax']), 'o',
                color=plt.cm.viridis(float(r['alpha'])/2.0), ms=3, alpha=0.7)
    ax.plot([0,1], [0,1], 'k-', lw=0.8, alpha=0.5, label='Perfect agreement')
    ths = [float(r['theory']) for r in rows if int(r['L'])==128]
    nums = [float(r['S_over_Smax']) for r in rows if int(r['L'])==128]
    if ths:
        z = np.polyfit(ths, nums, 1)
        ax.plot(np.linspace(0,max(ths),100), np.polyval(z, np.linspace(0,max(ths),100)),
                'r--', lw=0.8, label=f'$y={z[0]:.2f}x{z[1]:+.2f}$')
    ax.set_xlabel('Mean-field prediction'); ax.set_ylabel(r'Numerical $S/S_{\max}$')
    ax.set_title(r'(d) Theory vs simulation ($L=128$)', fontsize=9)
    ax.legend(fontsize=7)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, 2))
    plt.colorbar(sm, ax=ax, shrink=0.85).set_label(r'$\alpha$', fontsize=8)

    plt.tight_layout(pad=0.5)
    plt.savefig('fig2.pdf'); plt.close()
    print('  → fig2.pdf')


def make_beta_L_main():
    """Main text: β(L) 3-panel figure."""
    rows = load('fss_near_critical.csv')
    if not rows: return
    groups = {}
    for r in rows:
        groups.setdefault((float(r['alpha']), float(r['p'])), []).append(r)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for ax_i, p_t in enumerate([0.5, 0.7, 0.9]):
        ax = axes[ax_i]
        for ci, alpha in enumerate([0.5, 0.7, 0.9, 1.0, 1.2]):
            key = (alpha, p_t)
            if key not in groups: continue
            grp = sorted(groups[key], key=lambda r: int(r['L']))
            Ls = [int(r['L']) for r in grp]; Ss = [float(r['S_mean']) for r in grp]
            betas = [np.log(max(.01,Ss[i])/max(.01,Ss[i-1]))/np.log(Ls[i]/Ls[i-1])
                     for i in range(1, len(Ls))]
            Lm = [np.sqrt(Ls[i-1]*Ls[i]) for i in range(1, len(Ls))]
            st = '-' if alpha < 1 else ('--' if alpha == 1 else ':')
            ax.plot(Lm, betas, f'o{st}', color=COLORS[ci], ms=5, lw=1.2,
                    label=rf'$\alpha={alpha}$')
        ax.axhline(1, color='gray', ls=':', alpha=.5, lw=.8)
        ax.axhline(0, color='gray', ls=':', alpha=.3, lw=.8)
        ax.set_xscale('log', base=2); ax.set_xlabel(r'$\sqrt{L_1 L_2}$')
        if ax_i == 0: ax.set_ylabel(r'Local $\beta(L)$')
        ax.set_title(f'$p = {p_t}$'); ax.set_ylim(-0.2, 1.5)
        ax.legend(fontsize=7, loc='lower left')
    plt.tight_layout()
    plt.savefig('fig_betaL_main.pdf'); plt.close()
    print('  → fig_betaL_main.pdf')


def make_phase_diagram_revised():
    """Main text: revised phase diagram at L=256."""
    rows = load('alpha_scan.csv')
    if not rows: return
    target_L = 256
    grid = {(float(r['alpha']), float(r['p'])): float(r['S_over_Smax'])
            for r in rows if int(r['L']) == target_L}
    if not grid:
        target_L = 128
        grid = {(float(r['alpha']), float(r['p'])): float(r['S_over_Smax'])
                for r in rows if int(r['L']) == target_L}
    A = sorted(set(k[0] for k in grid)); P = sorted(set(k[1] for k in grid))
    Z = np.array([[grid.get((a,p), 0) for p in P] for a in A])
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    im = ax.pcolormesh(P, A, Z, cmap='RdYlBu_r', shading='nearest', vmin=0, vmax=0.9)
    ax.axhline(1.0, color='white', ls='--', lw=1.5)
    try:
        cs = ax.contour(P, A, Z, levels=[0.1,0.3,0.5], colors='white', linewidths=0.6)
        ax.clabel(cs, fmt='%.1f', fontsize=6, colors='white')
    except: pass
    ax.text(0.5, 1.6, 'Area law\n(MIPT exists)', fontsize=7, ha='center',
            color='white', weight='bold')
    ax.text(0.5, 0.4, 'Volume law\n(no MIPT)', fontsize=7, ha='center',
            color='white', weight='bold')
    ax.set_xlabel('Measurement rate $p$'); ax.set_ylabel(r'Power-law exponent $\alpha$')
    ax.set_title(rf'Phase diagram ($L={target_L}$)')
    plt.colorbar(im, ax=ax, shrink=0.85).set_label(r'$S/S_{\max}$', fontsize=8)
    plt.tight_layout()
    plt.savefig('fig_phase_diagram_revised.pdf'); plt.close()
    print('  → fig_phase_diagram_revised.pdf')


# ================================================================
# SUPPLEMENT FIGURES
# ================================================================

def make_figS1():
    """Fig S1: ΔS rate-equation validation."""
    rows = load('delta_s_validation.csv')
    if not rows: return
    groups = {}
    for r in rows:
        groups.setdefault((float(r['alpha']), float(r['p'])), []).append(r)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    ax = axes[0]; ci = 0
    for (alpha, p), grp in sorted(groups.items()):
        if alpha > 0.01: continue
        s_c = [float(r['s_center']) for r in grp]
        ds = [float(r['ds_gate_cross']) for r in grp]
        cnt = [int(r['cnt_gc']) for r in grp]
        mask = [c > 20 for c in cnt]
        ax.plot([1-s for s, m in zip(s_c, mask) if m],
                [d for d, m in zip(ds, mask) if m],
                'o-', color=COLORS[ci], ms=4, label=f'p={p:.1f}'); ci += 1
    x = np.linspace(0, 1, 50)
    ax.plot(x, x*0.5, 'k--', alpha=0.5, label=r'$\propto (1-s)$')
    ax.set_xlabel(r'$1 - S/S_{\max}$')
    ax.set_ylabel(r'$\langle\Delta S\,|\,\mathrm{gate\ crosses}\rangle$')
    ax.set_title(r'(a) Gate crossing: $\alpha=0$, $L=64$'); ax.legend(fontsize=6)

    ax = axes[1]; ci = 0
    for (alpha, p), grp in sorted(groups.items()):
        if alpha > 0.01: continue
        s_c = [float(r['s_center']) for r in grp]
        ds = [float(r['ds_meas']) for r in grp]
        cnt = [int(r['cnt_m']) for r in grp]
        mask = [c > 20 for c in cnt]
        ax.plot([s for s, m in zip(s_c, mask) if m],
                [d for d, m in zip(ds, mask) if m],
                's-', color=COLORS[ci], ms=4, label=f'p={p:.1f}'); ci += 1
    ax.plot(x, -x*0.3, 'k--', alpha=0.5, label=r'$\propto -s$')
    ax.set_xlabel(r'$S/S_{\max}$')
    ax.set_ylabel(r'$\langle\Delta S\,|\,\mathrm{measurement}\rangle$')
    ax.set_title(r'(b) Measurement: $\alpha=0$, $L=64$'); ax.legend(fontsize=6)

    plt.tight_layout()
    plt.savefig('figS1_delta_s.pdf'); plt.close()
    print('  → figS1_delta_s.pdf')


def make_figS2():
    """Fig S2/S3: finite-size scaling β(L)."""
    rows = load('fss_near_critical.csv')
    if not rows: return
    groups = {}
    for r in rows:
        groups.setdefault((float(r['alpha']), float(r['p'])), []).append(r)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    for ax_i, p_t in enumerate([0.5, 0.7]):
        ax = axes[ax_i]
        for ci, alpha in enumerate([0.5, 0.7, 0.9, 1.0, 1.1, 1.2]):
            key = (alpha, p_t)
            if key not in groups: continue
            grp = sorted(groups[key], key=lambda r: int(r['L']))
            Ls = [int(r['L']) for r in grp]; Ss = [float(r['S_mean']) for r in grp]
            betas = [np.log(max(.01,Ss[i])/max(.01,Ss[i-1]))/np.log(Ls[i]/Ls[i-1])
                     for i in range(1, len(Ls))]
            Lm = [np.sqrt(Ls[i-1]*Ls[i]) for i in range(1, len(Ls))]
            st = '-' if alpha < 1 else ('--' if alpha == 1 else ':')
            ax.plot(Lm, betas, f'o{st}', color=COLORS[ci], ms=5, lw=1.2,
                    label=rf'$\alpha={alpha}$')
        ax.axhline(1, color='gray', ls=':', alpha=.5)
        ax.axhline(0, color='gray', ls=':', alpha=.3)
        ax.set_xscale('log', base=2); ax.set_xlabel(r'$\sqrt{L_1 L_2}$')
        ax.set_ylabel(r'Local $\beta(L)$')
        ax.set_title(f'({"a" if ax_i==0 else "b"}) $p = {p_t}$')
        ax.set_ylim(-0.1, 1.4); ax.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig('figS2_fss.pdf'); plt.close()
    print('  → figS2_fss.pdf')


def make_figS3():
    """Fig S4: marginal α=1."""
    rows = load('marginal_alpha1.csv')
    if not rows: return
    groups = {}
    for r in rows:
        groups.setdefault(float(r['p']), []).append(r)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    ax = axes[0]
    for ci, p in enumerate(sorted(groups)):
        grp = sorted(groups[p], key=lambda r: int(r['L']))
        Ls = [int(r['L']) for r in grp]; Ss = [float(r['S_mean']) for r in grp]
        errs = [float(r['S_err']) for r in grp]
        ax.errorbar(Ls, Ss, yerr=errs, fmt='o-', color=COLORS[ci], ms=4,
                    label=f'p={p:.1f}')
    ax.set_xscale('log', base=2); ax.set_yscale('log', base=2)
    ax.set_xlabel('$L$'); ax.set_ylabel('$S$')
    ax.set_title(r'(a) $\alpha=1$: $S$ vs $L$'); ax.legend(fontsize=6)

    ax = axes[1]
    for ci, p in enumerate(sorted(groups)):
        grp = sorted(groups[p], key=lambda r: int(r['L']))
        Ls = [int(r['L']) for r in grp]
        vals = [float(r['S_lnL_over_L']) for r in grp]
        ax.plot(Ls, vals, 'o-', color=COLORS[ci], ms=4, label=f'p={p:.1f}')
    ax.set_xscale('log', base=2); ax.set_xlabel('$L$')
    ax.set_ylabel(r'$S \cdot \ln L / L$')
    ax.set_title(r'(b) $\alpha=1$: test for $S \sim L/\ln L$'); ax.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig('figS3_marginal.pdf'); plt.close()
    print('  → figS3_marginal.pdf')


def make_figS4():
    """Fig S5: Haar-random universality check."""
    rows = load('haar_check.csv')
    if not rows: return
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    for ax_i, a_t in enumerate([0.0, 2.0]):
        ax = axes[ax_i]
        Ls_s = sorted(set(int(r['L']) for r in rows
                          if abs(float(r['alpha'])-a_t) < 0.01))
        for ci, L in enumerate(Ls_s):
            grp = sorted([r for r in rows if int(r['L'])==L
                          and abs(float(r['alpha'])-a_t)<0.01],
                         key=lambda r: float(r['p']))
            ps = [float(r['p']) for r in grp]; ss = [float(r['S_over_Smax']) for r in grp]
            ax.plot(ps, ss, 'o-', color=C_SIZE[ci], ms=4, lw=1, label=f'$L={L}$')
        if a_t < 0.01:
            pt = np.linspace(0.01, 0.99, 100)
            ax.plot(pt, (1-pt)*0.5/((1-pt)*0.5+pt), 'k--', alpha=0.4, label='Mean field')
        ax.set_xlabel('$p$'); ax.set_ylabel(r'$S/S_{\max}$')
        ax.set_title(f'({"a" if ax_i==0 else "b"}) Haar, $\\alpha={a_t:.0f}$')
        ax.legend(fontsize=6); ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig('figS4_haar.pdf'); plt.close()
    print('  → figS4_haar.pdf')


def make_figS5():
    """Fig S6: thermalization convergence."""
    rows = load('thermalization_check.csv')
    if not rows: return
    groups = {}
    for r in rows:
        groups.setdefault((float(r['alpha']), int(r['L']), float(r['p'])), []).append(r)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for ci, (key, grp) in enumerate(sorted(groups.items())):
        alpha, L, p = key
        grp = sorted(grp, key=lambda r: int(r['depth']))
        depths = [int(r['depth'])/L for r in grp]; Ss = [float(r['S_mean']) for r in grp]
        errs = [float(r['S_err']) for r in grp]
        ax.errorbar(depths, Ss, yerr=errs, fmt='o-', color=COLORS[ci%len(COLORS)],
                    ms=3, lw=1, label=rf'$\alpha={alpha}$, L={L}, p={p}')
    ax.set_xlabel('Depth / $L$'); ax.set_ylabel(r'$\langle S \rangle$')
    ax.set_title('Thermalization convergence'); ax.legend(fontsize=5, ncol=2)
    plt.tight_layout()
    plt.savefig('figS5_thermalization.pdf'); plt.close()
    print('  → figS5_thermalization.pdf')


def make_figS7():
    """Fig S7: free-fermion Hamiltonian comparison (NEW)."""
    rows = load('free_fermion_mipt_gpu.csv')
    if not rows:
        print('  [SKIP] figS7 — missing free_fermion_mipt_gpu.csv'); return

    # Build lookup: (alpha, gamma, L) -> (S_mean, S_err)
    data = {}
    for r in rows:
        key = (float(r['alpha']), float(r['gamma']), int(r['L']))
        data[key] = (float(r['S_mean']), float(r['S_err']))

    alphas = sorted(set(k[0] for k in data))
    Ls = sorted(set(k[2] for k in data))

    # Circuit β from paper Table I (p=0.5 and p=0.7)
    ac = [0.0, 0.5, 0.9, 1.0, 1.5, 2.0]
    bc_p05     = [1.05, 1.04, 0.94, 0.90, 0.58, 0.31]
    bc_p05_err = [0.02, 0.02, 0.01, 0.02, 0.03, 0.05]

    # Compute Hamiltonian β(128→256) at γ=0.2
    bf, bf_err = [], []
    for a in alphas:
        k1, k2 = (a, 0.2, 128), (a, 0.2, 256)
        if k1 in data and k2 in data:
            s1, e1 = data[k1]; s2, e2 = data[k2]
            S1, S2 = s1/(128/2)*128/2, s2/(256/2)*256/2  # actual entropy
            S1 = data[k1][0]; S2 = data[k2][0]  # S_mean directly
            # S/Smax -> S = S/Smax * L/2
            S1_abs = data[k1][0]  # This is S_mean, not S/Smax
            S2_abs = data[k2][0]
            # Check if CSV has S_mean or S_over_Smax... need to check
            # From the code, S_mean is the actual entropy value
            if S1_abs > 0.05 and S2_abs > 0.05:
                b = np.log(S2_abs/S1_abs) / np.log(256/128)
                # error propagation
                dS1 = data[k1][1]; dS2 = data[k2][1]
                db = (1/np.log(2)) * np.sqrt((dS1/S1_abs)**2 + (dS2/S2_abs)**2)
                bf.append(b); bf_err.append(db)
            else:
                bf.append(np.nan); bf_err.append(np.nan)
        else:
            bf.append(np.nan); bf_err.append(np.nan)

    # γ=0.5 S/Smax for panel (b)
    g05 = {}
    for L in Ls:
        g05[L] = []
        for a in alphas:
            k = (a, 0.5, L)
            if k in data:
                s_over = float([r for r in rows if abs(float(r['alpha'])-a)<0.01
                                and abs(float(r['gamma'])-0.5)<0.01
                                and int(r['L'])==L][0]['S_over_Smax'])
                g05[L].append(s_over)
            else:
                g05[L].append(np.nan)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # Panel (a): β comparison
    ax1.errorbar(ac, bc_p05, yerr=bc_p05_err, fmt='o-', color='#3266ad',
                 lw=1.8, ms=6, capsize=3, label=r'Circuit ($p=0.5$)', zorder=5)
    valid = [(a, b, e) for a, b, e in zip(alphas, bf, bf_err) if not np.isnan(b)]
    if valid:
        aa, bb, ee = zip(*valid)
        ax1.errorbar(list(aa), list(bb), yerr=list(ee), fmt='D-', color='#D85A30',
                     lw=1.8, ms=5, capsize=3,
                     label=r'Hamiltonian + Kac ($\gamma\!=\!0.2$)', zorder=5)
    ax1.axhline(1.0, color='gray', ls=':', lw=0.7, alpha=0.4)
    ax1.axhline(0.0, color='gray', ls=':', lw=0.7, alpha=0.4)
    ax1.axvline(1.0, color='gray', ls='--', lw=0.8, alpha=0.3)
    ax1.set_xlabel(r'$\alpha$'); ax1.set_ylabel(r'$\beta$ ($S \sim L^\beta$)')
    ax1.set_xlim(-0.15, 2.2); ax1.set_ylim(-0.2, 1.3)
    ax1.legend(loc='center left', framealpha=0.9, fontsize=7.5)
    ax1.set_title(r'(a) Scaling exponent', fontsize=10)

    # Panel (b): S/Smax vs α
    c4 = ['#e74c3c', '#e67e22', '#27ae60', '#2980b9']
    for ci, L in enumerate([L for L in Ls if L >= 32]):
        if L not in g05: continue
        ax2.plot(alphas, g05[L], 'osD v'[ci%4]+'-', ms=4, lw=1.2, color=c4[ci%4],
                 label=f'$L={L}$')
    ax2.axvline(1.0, color='gray', ls='--', lw=0.8, alpha=0.3)
    ax2.set_xlabel(r'$\alpha$'); ax2.set_ylabel(r'$S/S_{\max}$')
    ax2.set_xlim(-0.15, 2.2)
    ax2.legend(loc='upper left', framealpha=0.9, fontsize=7.5)
    ax2.set_title(r'(b) Hamiltonian $S/S_{\max}$ ($\gamma\!=\!0.5$)', fontsize=10)
    plt.tight_layout(w_pad=1.5)
    plt.savefig('figS7_hamiltonian.pdf'); plt.close()
    print('  → figS7_hamiltonian.pdf')


def make_figS8():
    """Fig S8: gate ensemble comparison."""
    rows = load('gate_ensemble_check.csv')
    if not rows: return
    names = {1: "Original (CNOT+CZ)", 2: "CNOT-only", 3: "CNOT+SWAP"}
    markers = {1: 'o', 2: 's', 3: '^'}
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for gv in [1, 2, 3]:
        grp = sorted([r for r in rows if int(r['gate_variant'])==gv],
                     key=lambda r: float(r['p']))
        ps = [float(r['p']) for r in grp]; ss = [float(r['S_over_Smax']) for r in grp]
        errs = [float(r['S_err'])/32.0 for r in grp]
        ax.errorbar(ps, ss, yerr=errs, fmt=f'{markers[gv]}-', color=COLORS[gv-1],
                    ms=4, lw=1, label=names[gv])
    p_arr = np.linspace(0.01, 0.99, 100)
    ax.plot(p_arr, (1-p_arr)*0.5/((1-p_arr)*0.5+p_arr), 'k--', alpha=0.4,
            lw=0.8, label='Mean field')
    ax.set_xlabel('$p$'); ax.set_ylabel(r'$S/S_{\max}$')
    ax.set_title(r'Gate ensemble comparison ($\alpha=0$, $L=64$)')
    ax.legend(fontsize=6); ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig('figS6_gate_ensemble.pdf'); plt.close()
    print('  → figS6_gate_ensemble.pdf')


def print_table_I():
    """Print revised Table I to console."""
    rows = load('beta_table.csv')
    if not rows: return
    print("\n  === TABLE I: β ± δβ (L = 32–256) ===")
    print(f"  {'α':>4s}  {'p=0.1':>10s}  {'p=0.3':>10s}  {'p=0.5':>10s}  {'p=0.7':>10s}  {'p=0.9':>10s}")
    by_a = {}
    for r in rows:
        by_a.setdefault(float(r['alpha']), {})[float(r['p'])] = (
            float(r['beta']), float(r['beta_err']))
    for a in sorted(by_a):
        parts = []
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            if p in by_a[a]:
                b, e = by_a[a][p]
                parts.append(f"{b:.2f}({int(round(e*100)):d})")
            else:
                parts.append("---")
        print(f"  {a:4.1f}  " + "  ".join(f"{s:>10s}" for s in parts))


# ================================================================
# DISPATCH
# ================================================================
if __name__ == '__main__':
    do_main = '--main' in sys.argv or not any(
        a in sys.argv for a in ['--main', '--supplement'])
    do_supp = '--supplement' in sys.argv or not any(
        a in sys.argv for a in ['--main', '--supplement'])

    print("=" * 55)
    print("  Generating publication figures")
    print(f"  Data: {DATADIR}/")
    print("=" * 55)

    if do_main:
        print("\n--- Main text ---")
        make_fig1()
        make_fig2()
        make_beta_L_main()
        make_phase_diagram_revised()
        print_table_I()

    if do_supp:
        print("\n--- Supplement ---")
        make_figS1()       # S1: ΔS validation
        make_figS2()       # S2/S3: finite-size scaling
        make_figS3()       # S4: marginal α=1
        make_figS4()       # S5: Haar check
        make_figS5()       # S6: thermalization
        make_figS7()       # S7: free-fermion Hamiltonian [NEW]
        make_figS8()       # S8: gate ensemble

    print("\n" + "=" * 55)
    print("  Done.")
    print("=" * 55)
