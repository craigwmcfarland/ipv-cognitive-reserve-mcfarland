"""
---
title: "Figure S2: predicted delayed recall across cognitive reserve and PCG tNAA/tCr"
output: figures/figureS2.png
---
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path
from scipy.spatial import ConvexHull

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
plt.rcParams.update({'font.family': FONT, 'mathtext.fontset': 'custom', 'mathtext.rm': FONT,
                     'mathtext.it': FONT + ':italic', 'mathtext.bf': FONT + ':bold', 'axes.linewidth': 1.06})
PCG = 'naa_naag_crpcr_pcg'
COV = ['log_ipv_bi_capped', 'age', 'education_years', 'ptsd_pcl5']
q = d.dropna(subset=[Y, RES, PCG] + COV)
t_raw, m_raw, y_raw = q[RES].values, q[PCG].values, q[Y].values
tz, mz = z(t_raw), z(m_raw)
b = ols(z(y_raw), [tz, mz, tz * mz] + [z(q[c]) for c in COV])[0]

gt = np.linspace(tz.min(), tz.max(), 300); gm = np.linspace(mz.min(), mz.max(), 300)
T, Mg = np.meshgrid(gt, gm)
Z = b[0] + b[1] * T + b[2] * Mg + b[3] * T * Mg
hull = ConvexHull(np.column_stack([tz, mz]))
inside = Path(np.column_stack([tz, mz])[hull.vertices]).contains_points(np.column_stack([T.ravel(), Mg.ravel()])).reshape(T.shape)
Z = np.where(inside, Z, np.nan)

to_raw_t = lambda v: v * t_raw.std(ddof=1) + t_raw.mean()
to_raw_m = lambda v: v * m_raw.std(ddof=1) + m_raw.mean()
cmap = LinearSegmentedColormap.from_list('s2', ['#29216A', '#1456D2', '#0B98CE', '#25B3AD', '#8DBF75', '#E1C04C', '#FAEE17'])

fig, ax = plt.subplots(figsize=(6.4, 5.0))
lim = np.nanmax(np.abs(Z))
pc = ax.pcolormesh(to_raw_t(T), to_raw_m(Mg), Z, cmap=cmap, vmin=-lim, vmax=lim, shading='auto')
cs = ax.contour(to_raw_t(T), to_raw_m(Mg), Z, levels=np.arange(-2, 2.01, 0.5), colors='white', linewidths=0.8)
ax.clabel(cs, fmt='%.1f', fontsize=8)
sc = ax.scatter(t_raw, m_raw, c=z(y_raw), cmap=cmap, vmin=-lim, vmax=lim, s=34, edgecolors='black', linewidths=0.8, zorder=3)
for v in (-1, 1):
    ax.axhline(to_raw_m(v), color='black', linewidth=0.9, linestyle='--', zorder=2)
ax.text(to_raw_t(gt.max()), to_raw_m(-1), '−1 SD', ha='right', va='bottom', fontsize=9, color='#E00000')
ax.text(to_raw_t(gt.max()), to_raw_m(1), '+1 SD', ha='right', va='bottom', fontsize=9, color='#E00000')
cb = fig.colorbar(pc, ax=ax, pad=0.02); cb.set_label('Delayed recall (SD)', fontsize=11, fontweight='bold')
ax.set_xlabel('Cognitive reserve (ToPF standard score)', fontsize=12, fontweight='bold')
ax.set_ylabel('PCG tNAA/tCr', fontsize=12, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
OUT = os.path.join(FIGURES, 'figureS2.png')
fig.savefig(OUT, dpi=300, facecolor='white')
print('  -> figure written to figures/figureS2.png | n =', len(q))
