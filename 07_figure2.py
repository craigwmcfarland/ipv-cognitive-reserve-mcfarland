"""
---
title: "Figure 2: cognitive reserve and memory by injury burden"
output: figures/figure2.png
---
"""
import os
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.font_manager as fm, matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(HERE, 'data', 'IPV_CognitiveReserve_DATA.csv'))
FIGURES = os.path.join(HERE, 'figures'); os.makedirs(FIGURES, exist_ok=True)
OUT = os.path.join(FIGURES, 'figure2.png')
FONT = 'Arial' if any('Arial' in f.name for f in fm.fontManager.ttflist) else 'Liberation Sans'
plt.rcParams.update({'font.family': FONT, 'mathtext.fontset': 'custom', 'mathtext.rm': FONT,
                     'mathtext.it': FONT + ':italic', 'mathtext.bf': FONT + ':bold', 'axes.linewidth': 1.06, 'xtick.major.width': 1.06,
                     'ytick.major.width': 1.06, 'xtick.major.size': 6, 'ytick.major.size': 6})


GREY_F, GREY_E = '#B9B7C0', '#3E3A4C'
BLUE_F, BLUE_E = '#0F5EDC', '#0A3E92'
S = dict(hi=(BLUE_F, BLUE_E), lo=('#4A4656', '#2B2833'), band_hi='#CBD9F6', band_lo='#E4E3E7',
                 none=('#D9D7DE', GREY_E), mid=('#8FB0EC', '#3D6CC0'),
                 c_ns=('#C9C7CF', '#6B6878'), c_trend=('#7FA6EA', '#3D6CC0'), c_sig=(BLUE_F, BLUE_E))

Y, RES, MEMCOV = 'np_ldfr_scaled', 'np_topf_ss', ['education_years', 'ptsd_pcl5']
q = d.dropna(subset=[Y, RES, 'ipv_bi_count'])
hi_m = q.ipv_bi_count >= 10

def z(v): v = np.asarray(v, float); return (v - v.mean()) / v.std(ddof=1)
def ols_hc3(y, cols):
    X = np.column_stack([np.ones(len(y))] + cols); Xi = np.linalg.pinv(X.T @ X); b = Xi @ X.T @ y
    r = y - X @ b; h = np.einsum('ij,jk,ik->i', X, Xi, X)
    V = Xi @ (X.T * ((r / (1 - h)) ** 2)) @ X @ Xi; df = len(y) - X.shape[1]
    return b, V, df


W, H, DPI = 8.9, 6.06, 300
fig = plt.figure(figsize=(W, H))
axA = fig.add_axes([0.075, 0.095, 0.555, 0.86])
axB = fig.add_axes([0.73, 0.575, 0.255, 0.38])
axC = fig.add_axes([0.73, 0.14, 0.255, 0.34])
TICK, LABEL, LETTER = 11, 13, 19


for sel, (fc, ec), band, size, z_ in ((~hi_m, S['lo'], S['band_lo'], 46, 3), (hi_m, S['hi'], S['band_hi'], 90, 4)):
    s = q[sel]; x, y = s[RES].values, s[Y].values
    X = np.column_stack([np.ones_like(x), x]); b = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ b; df = len(y) - 2; s2 = res @ res / df; XtXi = np.linalg.inv(X.T @ X)
    xs = np.linspace(x.min(), x.max(), 100); Xs = np.column_stack([np.ones_like(xs), xs])
    se = np.sqrt(s2 * np.einsum('ij,jk,ik->i', Xs, XtXi, Xs)); tc = stats.t.ppf(.975, df)
    axA.fill_between(xs, Xs @ b - tc * se, Xs @ b + tc * se, color=band, alpha=0.75, zorder=1, linewidth=0)
    axA.plot(xs, Xs @ b, color=fc if sel is hi_m else ec, linewidth=2.6, zorder=2)
    axA.scatter(x, y, s=size, c=fc, edgecolors=ec, linewidths=1.0, zorder=z_)
r_hi = stats.spearmanr(q[hi_m][RES], q[hi_m][Y]); r_lo = stats.spearmanr(q[~hi_m][RES], q[~hi_m][Y])
print(f'Panel A  >=10 injuries: rho = {r_hi.statistic:.2f}, P = {r_hi.pvalue:.3f}, n = {hi_m.sum()}'
      f' | <10 injuries: rho = {r_lo.statistic:.2f}, P = {r_lo.pvalue:.3f}, n = {(~hi_m).sum()}')
axA.set_xlabel('Cognitive reserve', fontsize=LABEL + 2, fontweight='bold')
axA.set_ylabel(r'$\bf{Delayed\ recall}$ (scaled score)', fontsize=LABEL + 2)
axA.set_xticks([80, 90, 100, 110, 120]); axA.set_yticks([4, 6, 8, 10, 12, 14, 16])
axA.set_ylim(2.7, 17.4); axA.tick_params(labelsize=TICK + 1)
axA.spines[['top', 'right']].set_visible(False)
leg = axA.legend(handles=[Patch(facecolor=S['hi'][0], edgecolor=S['hi'][1], label='Highest neurotrauma burden (≥10)'),
                          Patch(facecolor=S['lo'][0], edgecolor=S['lo'][1], label='Lower neurotrauma burden (<10)')],
                 title='Exposure group:', loc='upper left', fontsize=12, title_fontsize=12, frameon=True,
                 edgecolor='black', framealpha=1, borderpad=0.6)
leg.get_frame().set_linewidth(1.2); leg.get_title().set_fontweight('bold'); leg._legend_box.align = 'left'


bands = [('None', q.ipv_bi_count == 0, S['none']), ('1-9', (q.ipv_bi_count > 0) & (q.ipv_bi_count < 10), S['mid']),
         ('≥10', hi_m, S['hi'])]
for i, (lab, sel, (fc, ec)) in enumerate(bands):
    s = q[sel]; rr = stats.spearmanr(s[RES], s[Y]); r = rr.statistic; se = 1 / np.sqrt(len(s) - 3)
    lo, hi = np.tanh(np.arctanh(r) - 1.96 * se), np.tanh(np.arctanh(r) + 1.96 * se)
    axB.bar(i, r, width=0.72, color=fc, edgecolor=ec, linewidth=1.6, zorder=2)
    axB.errorbar(i, r, yerr=[[r - lo], [hi - r]], fmt='none', ecolor='black', elinewidth=1.4, capsize=7, capthick=1.4, zorder=3)
    star = '***' if rr.pvalue < .001 else '**' if rr.pvalue < .01 else '*' if rr.pvalue < .05 else ''
    if star: axB.text(i, hi + 0.02, star, ha='center', fontsize=15, fontweight='bold')
axB.axhline(0, color='black', linewidth=1.2, zorder=1)
axB.set_xticks([0, 1, 2]); axB.set_xticklabels(['None', '1-9', '≥10'], fontsize=TICK)
axB.set_yticks([-0.5, 0, 0.5, 1.0]); axB.set_yticklabels(['−0.5', '0.0', '0.5', '1.0']); axB.set_ylim(-0.7, 1.2); axB.tick_params(labelsize=TICK)
axB.set_ylabel('Reserve–memory ρ', fontsize=LABEL - 1, fontweight='bold')
axB.set_xlabel('Brain injuries', fontsize=LABEL, fontweight='bold')
axB.spines[['top', 'right']].set_visible(False)


for k in range(5, 13):
    s = q.dropna(subset=MEMCOV); t = z(s[RES]); h = (s.ipv_bi_count >= k).astype(float).values
    b, V, df = ols_hc3(z(s[Y]), [t, h, t * h] + [z(s[c]) for c in MEMCOV])
    p = 2 * stats.t.sf(abs(b[3] / np.sqrt(V[3, 3])), df)
    fc, ec = S['c_sig'] if p < .05 else S['c_trend'] if p < .10 else S['c_ns']
    axC.bar(k, b[3], width=0.88, color=fc, edgecolor=ec, linewidth=1.6, zorder=2)
    if p < .05: axC.text(k, b[3] + 0.03, '*', ha='center', fontsize=13, fontweight='bold')
axC.set_xticks([5, 7, 9, 11]); axC.set_yticks([0, 0.5, 1.0]); axC.set_ylim(0, 1.22); axC.tick_params(labelsize=TICK)
axC.set_xlim(4.4, 12.6)
axC.set_ylabel('Reserve × injury β', fontsize=LABEL - 1, fontweight='bold')
axC.set_xlabel('Highest neurotrauma burden\nthreshold', fontsize=LABEL - 1, fontweight='bold')
axC.spines[['top', 'right']].set_visible(False)

for lab, x, y in (('A', 0.005, 0.955), ('B', 0.655, 0.955), ('C', 0.655, 0.49)):
    fig.text(x, y, lab, fontsize=LETTER + 6, fontweight='bold')
fig.savefig(OUT, dpi=DPI, facecolor='white')
print('  -> figure written to figures/figure2.png')
