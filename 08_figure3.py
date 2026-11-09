"""
---
title: "Figure 3: cognitive reserve, PCG tNAA/tCr and memory"
output: figures/figure3.png
---
"""
import os
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.font_manager as fm, matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data', 'IPV_CognitiveReserve_DATA.csv')
FIGURES = os.path.join(HERE, 'figures'); os.makedirs(FIGURES, exist_ok=True)
OUT = os.path.join(FIGURES, 'figure3.png')
FONT = 'Arial' if any('Arial' in f.name for f in fm.fontManager.ttflist) else 'Liberation Sans'
plt.rcParams.update({'font.family': FONT, 'mathtext.fontset': 'custom',
    'mathtext.rm': FONT, 'mathtext.it': FONT + ':italic', 'mathtext.bf': FONT + ':bold',
    'axes.linewidth': 1.06, 'xtick.major.width': 1.06, 'ytick.major.width': 1.06,
    'xtick.major.size': 6, 'ytick.major.size': 6})

TICK_A, LABEL_A = 15, 16
TICK, LABEL = 13, 14
LETTER, STAR = 21, 17
BLUE, BLUE_EDGE, GREY, GREY_EDGE = '#0F5EDC', '#0A3E92', '#8A8598', '#3E3A4C'
CMAP_STOPS = [
    '#29216A','#303092','#2B379D','#273DA7','#2243B1','#1C4BBF','#1850C7','#1456D2',
    '#0F5CDC','#1061DC','#1066DB','#116ADA','#126FD9','#1274D8','#1378D7','#147DD7',
    '#1382D6','#1287D4','#108BD3','#0E90D1','#0C94D0','#0B98CE','#099DCC','#07A1CB',
    '#08A5C8','#0DA7C4','#12AABF','#17ACBA','#1DAFB4','#21B1B1','#25B3AD','#2BB6A7',
    '#36B89F','#41B99A','#4CBA95','#57BB90','#63BC8A','#6EBD84','#79BE7F','#83BF79',
    '#8DBF75','#96BE71','#A0BE6D','#A9BD69','#B2BD65','#BCBC62','#C5BC5E','#CDBB5A',
    '#D5BC56','#DBBE51','#E1C04C','#E6C147','#EBC342','#F4C53B','#F9C636','#FEC932',
    '#FDCF2E','#FDD529','#FCDB25','#FCE120','#FBE81C','#FAEE17','#FAF513','#C5C60B']
cmap = LinearSegmentedColormap.from_list('fig3', CMAP_STOPS)
VMIN, VMAX = -0.65, 0.65

d = pd.read_csv(DATA)
COV = ['log_ipv_bi_capped', 'age', 'education_years', 'ptsd_pcl5']
def z(v): v = np.asarray(v, float); return (v - v.mean()) / v.std(ddof=1)
class Fit:
    def __init__(s, y, X):
        s.n, s.k = X.shape; Xi = np.linalg.pinv(X.T @ X); s.beta = Xi @ X.T @ y
        r = y - X @ s.beta; h = np.einsum('ij,jk,ik->i', X, Xi, X)
        s.cov = Xi @ (X.T * ((r / (1 - h)) ** 2)) @ X @ Xi; s.df = s.n - s.k
def metab_fit(col):
    q = d[['np_ldfr_scaled', 'np_topf_ss', col] + COV].dropna()
    t = z(q.np_topf_ss); m = z(q[col])
    X = np.column_stack([np.ones(len(q)), t, m, t * m] + [z(q[c]) for c in COV])
    return q, t, m, Fit(z(q.np_ldfr_scaled), X)


def est(F, w):
    w = np.asarray(w, float); b = w @ F.beta; se = np.sqrt(w @ F.cov @ w); tc = stats.t.ppf(.975, F.df)
    return b, b - tc * se, b + tc * se, 2 * stats.t.sf(abs(b / se), F.df)
VOXELS, METABS = ('pcg', 'dlpfc', 'coro'), ('naa_naag', 'glu_gln', 'mi', 'pch_gpc')
GRID = {(v, m_): est(metab_fit(f'{m_}_crpcr_{v}')[3], np.eye(4 + len(COV))[3]) for v in VOXELS for m_ in METABS}
FAM1 = [(v, 'naa_naag') for v in VOXELS]; FAM2 = [k for k in GRID if k not in FAM1]
QF = dict(zip(FAM1, stats.false_discovery_control([GRID[k][3] for k in FAM1])))
QF.update(zip(FAM2, stats.false_discovery_control([GRID[k][3] for k in FAM2])))
HEAT = np.array([[round(float(GRID[(v, m_)][0]), 3) for m_ in METABS] for v in VOXELS])
SIG = [(i, j) for i, v in enumerate(VOXELS) for j, m_ in enumerate(METABS) if QF[(v, m_)] < .05]
ROWS = ['PCG', 'DLPFC', 'Corona\nradiata']
COLS = ['tNAA/tCr', 'Glx/tCr', 'mI/tCr', 'Cho/tCr']

W, DPI = 8.05, 300
L, GAPX = 0.155, 0.145
WB, WC = 0.235, 0.305
SPAN = WB + GAPX + WC
gridh = (SPAN * W / 4) * 3
row2 = 2.35
GAP_A = 1.50
BM = 0.80
H = 0.60 + gridh + 0.72 + GAP_A + row2 + BM
XC = L + WB + GAPX
Y2, H2 = BM / H, row2 / H
Y1, H1 = (BM + row2 + GAP_A + 0.72) / H, gridh / H
fig = plt.figure(figsize=(W, H), dpi=DPI)
axA = fig.add_axes([L, Y1, SPAN, H1]); axB = fig.add_axes([L, Y2, WB, H2]); axC = fig.add_axes([XC, Y2, WC, H2])
CB_Y, CB_H = (Y1 * H - 0.685) / H, 0.226 / H
cax = fig.add_axes([L, CB_Y, SPAN, CB_H])

im = axA.imshow(HEAT, cmap=cmap, vmin=VMIN, vmax=VMAX, aspect='auto')
axA.set_xticks(range(4)); axA.set_yticks(range(3))
axA.set_xticklabels(COLS, fontsize=TICK_A, style='italic')
axA.set_yticklabels(ROWS, fontsize=TICK_A, fontweight='bold')
axA.tick_params(length=0); axA.tick_params(axis='x', pad=14); axA.tick_params(axis='y', pad=16)
for s in axA.spines.values(): s.set_visible(True)
for r, c in SIG:
    ins = (2.2 / 2) / ((SPAN * W / 4) * 72)
    axA.add_patch(Rectangle((c - 0.5 + ins, r - 0.5 + ins), 1 - 2 * ins, 1 - 2 * ins, fill=False,
                            edgecolor='white', linewidth=2.2, joinstyle='miter', capstyle='projecting', zorder=4))
    axA.text(c, r, '*', ha='center', va='center', color='white', fontsize=STAR + 9, fontweight='bold')
cb = fig.colorbar(im, cax=cax, orientation='horizontal')
cb.set_ticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6])
cb.ax.tick_params(labelsize=TICK_A, width=1.06, length=6); cb.outline.set_linewidth(1.06)
cb.set_label('Reserve × metabolite β', fontsize=LABEL_A, fontweight='bold', labelpad=8)


_FP = metab_fit('naa_naag_crpcr_pcg')[3]
_sl = [est(_FP, np.r_[0, 1, 0, -1, np.zeros(len(COV))]), est(_FP, np.r_[0, 1, 0, 1, np.zeros(len(COV))])]
est = [round(float(v[0]), 3) for v in _sl]; lo = [round(float(v[1]), 2) for v in _sl]; hi = [round(float(v[2]), 2) for v in _sl]
x = [0.34, 0.66]
axB.bar(x, est, width=0.20, color=[BLUE, GREY], edgecolor=[BLUE_EDGE, GREY_EDGE], linewidth=1.7, zorder=2)
for xi, e, l, h in zip(x, est, lo, hi):
    axB.errorbar(xi, e, yerr=[[e - l], [h - e]], fmt='none', ecolor='black', elinewidth=1.06, capsize=7, capthick=1.06, zorder=3)
axB.axhline(0, color='black', linewidth=0.81, zorder=1)
axB.set_xlim(0.15, 0.85); axB.set_ylim(-0.80, 1.68)
axB.set_xticks(x); axB.set_xticklabels(['Lower', 'Higher'], fontsize=TICK)
axB.set_yticks([-0.5, 0, 0.5, 1.0]); axB.set_yticklabels(['−0.5', '0.0', '0.5', '1.0'], fontsize=TICK)
axB.set_ylabel(r'$\bf{Memory\ advantage}$ (SD)', fontsize=LABEL, labelpad=8)
axB.set_xlabel(r'$\bf{PCG\ tNAA/tCr}$', fontsize=LABEL, labelpad=9)
axB.spines[['top', 'right']].set_visible(False)
axB.text(x[0], hi[0] + 0.07, '***', ha='center', va='bottom', fontsize=STAR, fontweight='bold')
yb = 1.52
axB.plot([x[0], x[0], x[1], x[1]], [yb - 0.06, yb, yb, yb - 0.06], color='black', linewidth=1.06,
         solid_joinstyle='miter', solid_capstyle='projecting')
axB.text(sum(x) / 2, yb + 0.02, '***', ha='center', va='bottom', fontsize=STAR, fontweight='bold')

q, t, m, F = metab_fit('naa_naag_crpcr_pcg')
yv = z(q.np_ldfr_scaled)
axC.scatter(t[m <= np.median(m)], yv[m <= np.median(m)], s=26, c=BLUE, edgecolors=BLUE_EDGE, linewidths=1.06, zorder=6, label='Lower')
axC.scatter(t[m > np.median(m)], yv[m > np.median(m)], s=26, c=GREY, edgecolors=GREY_EDGE, linewidths=1.06, zorder=5, label='Higher')
xs = np.linspace(t.min(), t.max(), 120); tc = stats.t.ppf(.975, F.df)
for m0, col, band, zb, zl in ((1, GREY_EDGE, '#E4E3E7', 1, 3), (-1, BLUE, '#CBD9F6', 2, 4)):
    c = np.column_stack([np.ones_like(xs), xs, np.full_like(xs, m0), xs * m0, np.zeros((len(xs), len(COV)))])
    pred = c @ F.beta; se = np.sqrt(np.einsum('ij,jk,ik->i', c, F.cov, c))
    axC.fill_between(xs, pred - tc * se, pred + tc * se, color=band, alpha=0.75 if m0 < 0 else 1, zorder=zb)
    axC.plot(xs, pred, color=col, linewidth=2.28, zorder=zl)
axC.set_xlabel(r'$\bf{Cognitive\ reserve}$ (SD)', fontsize=LABEL, labelpad=9)
axC.set_ylabel(r'$\bf{Delayed\ recall}$ (SD)', fontsize=LABEL, labelpad=8)
axC.set_xticks([-3, -2, -1, 0, 1]); axC.set_yticks([-3, -2, -1, 0, 1, 2]); axC.tick_params(labelsize=TICK)
axC.spines[['top', 'right']].set_visible(False)
leg = axC.legend(title='PCG tNAA/tCr:', fontsize=TICK - 5, title_fontsize=TICK - 4.5, loc='lower right',
                 frameon=True, edgecolor='black', framealpha=1, borderpad=0.3,
                 labelspacing=0.2, handletextpad=0.3, handlelength=0.8, markerscale=0.6)
leg.get_frame().set_linewidth(0.8); leg.get_title().set_fontweight('bold'); leg._legend_box.align = 'left'
LX_AB, LX_C = 0.030, XC - 0.112
for lab, lx, ly in (('A', LX_AB, Y1 + H1 + 0.003), ('B', LX_AB, Y2 + H2 + 0.003), ('C', LX_C, Y2 + H2 + 0.003)):
    fig.text(lx, ly, lab, ha='left', va='bottom', fontsize=LETTER, fontweight='bold')
fig.savefig(OUT, dpi=DPI, facecolor='white')
print('  -> figure written to figures/figure3.png | PCG model n =', F.n)
