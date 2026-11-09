"""
---
title: "Figure S1: model fit under alternative codings of injury exposure"
output: figures/figureS1.png
---
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(HERE, 'figures'); os.makedirs(FIGURES, exist_ok=True)
d = pd.read_csv(os.path.join(HERE, 'data', 'IPV_CognitiveReserve_DATA.csv'))
Y, RES = 'np_ldfr_scaled', 'np_topf_ss'


def z(v):
    v = np.asarray(v, float); return (v - v.mean()) / v.std(ddof=1)


def ols(y, cols):
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in cols]); y = np.asarray(y, float)
    b = np.linalg.pinv(X.T @ X) @ X.T @ y
    return b, y - X @ b, X.shape[1]

FONT = 'Arial' if any('Arial' in f.name for f in fm.fontManager.ttflist) else 'Liberation Sans'
plt.rcParams.update({'font.family': FONT, 'axes.linewidth': 1.2, 'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
                     'xtick.major.size': 5, 'ytick.major.size': 5})


q = d.dropna(subset=[Y, 'ipv_bi_count', 'education_years', 'ptsd_pcl5'])
yv = z(q[Y]); cv = [z(q.education_years), z(q.ptsd_pcl5)]; cc = q.ipv_bi_count.values.astype(float)


def aic(e):
    b, r, k = ols(yv, [e] + cv); n = len(yv); return n * np.log(float(r @ r) / n) + 2 * (k + 1)


A = {'logarithmic (primary)': aic(z(np.log1p(np.minimum(cc, 25)))), 'linear count': aic(z(np.minimum(cc, 25))),
     'ordinal (0 / 1–9 / ≥10)': aic(np.where(cc == 0, 0., np.where(cc < 10, 1., 2.))), 'binary any injury': aic((cc > 0).astype(float))}
best = min(A.values())
ORDER = [('binary any injury', '0 vs ≥1', '#B4B4BE', '#3A3A44'), ('linear count', 'Linear', '#2171B5', '#08519C'),
         ('ordinal (0 / 1–9 / ≥10)', 'Ordinal', '#9ECAE1', '#08519C'), ('logarithmic (primary)', 'log(1+X)', '#2171B5', '#08519C')]
DPI, W, H = 360, 1655, 1364
fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
ax = fig.add_axes([232 / W, (H - 1049) / H, (1622 - 232) / W, (1049 - 38) / H])
for i, (key, lab, fc, ec) in enumerate(ORDER):
    v = A[key] - best
    ax.bar(i, v, width=0.672, color=fc, edgecolor=ec, linewidth=1.5, zorder=2)
    ax.text(i, v + 0.17, f'{v:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.set_xlim(-0.513, 3.516); ax.set_ylim(0, 9.21)
ax.set_yticks([0, 2, 4, 6, 8]); ax.tick_params(axis='y', labelsize=12)
ax.set_xticks(range(4)); ax.set_xticklabels([o[1] for o in ORDER], rotation=30, ha='right', rotation_mode='anchor',
                                            fontsize=12, fontweight='bold')
ax.set_ylabel('ΔAIC from best model', fontsize=12.5, fontweight='bold', labelpad=6)
ax.set_xlabel('Exposure specification', fontsize=12.5, fontweight='bold', labelpad=6.8)
ax.spines[['top', 'right']].set_visible(False)
OUT = os.path.join(FIGURES, 'figureS1.png')
fig.savefig(OUT, dpi=DPI, facecolor='white')
print('  -> figure written to figures/figureS1.png | ΔAIC:', ', '.join(f'{o[1]} {A[o[0]] - best:.2f}' for o in ORDER))
