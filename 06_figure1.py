"""
---
title: "Figure 1: memory outcomes across injury burden"
output: figures/figure1.png, figures/figure1.pdf
---
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerBase

HERE = os.path.dirname(os.path.abspath(__file__))
D = pd.read_csv(os.path.join(HERE, 'data', 'IPV_CognitiveReserve_DATA.csv'))
FIGURES = os.path.join(HERE, 'figures'); os.makedirs(FIGURES, exist_ok=True)
OUT = os.path.join(FIGURES, 'figure1')
XTICK_A = 7.5
VIOLIN_ALPHA = 0.66

_avail = {f.name for f in fm.fontManager.ttflist}
FAM = next((n for n in ('Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans') if n in _avail), 'sans-serif')
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': [FAM],
 'mathtext.fontset': 'custom', 'mathtext.rm': FAM, 'mathtext.it': FAM, 'mathtext.bf': FAM + ':bold',
 'font.size': 7.5, 'axes.linewidth': 1.0, 'xtick.major.width': 1.0, 'ytick.major.width': 1.0,
 'xtick.major.size': 3.4, 'ytick.major.size': 3.4, 'xtick.direction': 'out', 'ytick.direction': 'out',
 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 400, 'pdf.fonttype': 42,
 'axes.edgecolor': '#000000', 'text.color': '#000000', 'axes.labelcolor': '#000000',
 'xtick.color': '#000000', 'ytick.color': '#000000', 'legend.frameon': False})

RB_F = ['#1F48C1', '#28A4C3', '#33CCA1', '#30CF46', '#85D42B', '#DEBE21']
RB_E = ['#18254E', '#1D4953', '#25604F', '#24612C', '#446322', '#675B1E']
GREY_F, GREY_E, RED = '#CCCCCC', '#8A8A8A', '#D62728'
NORM = [('np_ldfr_scaled', 'CVLT\ndelayed recall', 10, 3), ('cvlt_learn', 'CVLT\nverbal learning', 10, 3),
        ('np_dr_index', 'CVLT\nmemory index', 100, 15), ('composite_memory', 'CNSVS\nmemory composite', 100, 15),
        ('verbal_memory', 'CNSVS\nverbal memory', 100, 15), ('visual_memory', 'CNSVS\nvisual memory', 100, 15)]
MEAS_RAW = [('np_ldfr_scaled', 'Long delay free recall', 'Scaled score', (2, 18), 10),
            ('cvlt_learn', 'Verbal learning', 'Scaled score', (2, 18), 10),
            ('np_dr_index', 'Memory Index', 'Index score', (60, 140), 100),
            ('composite_memory', 'Memory composite', 'Standard score', (40, 160), 100),
            ('verbal_memory', 'Verbal memory', 'Standard score', (40, 160), 100),
            ('visual_memory', 'Visual memory', 'Standard score', (40, 160), 100)]

def rows(cols): return D.dropna(subset=cols)
def kde(v, xk):
    bw = 1.06 * v.std(ddof=1) * len(v) ** (-1 / 5)
    return np.mean(np.exp(-0.5 * ((xk[:, None] - v[None, :]) / bw) ** 2) / (bw * np.sqrt(2 * np.pi)), axis=1)

class _Sq(HandlerBase):
    def __init__(self, cols, frame): self.cols, self.frame = cols, frame; super().__init__()
    def create_artists(self, legend, orig, xd, yd, w, h, fs, trans):
        side = min(w, h); x0 = xd + (w - side) / 2.0; y0 = yd + (h - side) / 2.0 - 0.34 * fs; n = len(self.cols)
        out = [plt.Rectangle([x0 + q * side / n, y0], side / n + 0.4, side, facecolor=self.cols[q], edgecolor='none',
                             lw=0, transform=trans) for q in range(n)]
        out.append(plt.Rectangle([x0, y0], side, side, facecolor='none', edgecolor=self.frame, lw=1.0, transform=trans))
        return out
def _key(fig):
    k1 = Patch(facecolor='none', edgecolor='none'); k2 = Patch(facecolor='none', edgecolor='none')
    lg = fig.legend([k1, k2], ['None', 'Ten or more'], handler_map={k1: _Sq([GREY_F], GREY_E), k2: _Sq(RB_F, '#2A2A2A')},
                    loc='lower center', bbox_to_anchor=(0.5, 0.004), ncol=2, frameon=True, edgecolor='black',
                    framealpha=1.0, fancybox=False, fontsize=7.8, handlelength=1.15, handleheight=1.15,
                    handletextpad=.65, columnspacing=4.2, borderpad=.85)
    lg.set_title('Number of partner-inflicted brain injuries:', prop={'size': 7.8, 'weight': 'bold'})
    lg._legend_box.align = 'left'; lg._legend_box.sep = 3.0
    return lg

def fig1():
    FW1 = 7.30; GL, GR = 0.86, 0.86 + 2 * 2.22 + 1.56; SH = 3.90; ROWP = 2.16
    GRIDH = 1.04 + ROWP + 1.40; GAP = 0.88; FH1 = GRIDH + SH + GAP + 0.24
    fig = plt.figure(figsize=(FW1, FH1))
    axv = fig.add_axes([GL / FW1, (GRIDH + GAP) / FH1, (GR - GL) / FW1, (SH - 0.72) / FH1])
    data = [((rows([c])[c].to_numpy(float)) - mu) / sd for c, _, mu, sd in NORM]
    vp = axv.violinplot(data, positions=range(6), widths=.86, showextrema=False, showmedians=False)
    for i, b in enumerate(vp['bodies']):
        b.set_facecolor(RB_F[i]); b.set_edgecolor(RB_E[i]); b.set_alpha(VIOLIN_ALPHA); b.set_linewidth(.9)
    rng = np.random.default_rng(11)
    for i, v in enumerate(data):
        axv.scatter(i + rng.uniform(-.17, .17, len(v)), v, s=6.5, facecolor=RB_F[i], edgecolors=RB_E[i],
                    linewidths=.35, alpha=.78, zorder=4)
    bp = axv.boxplot(data, positions=range(6), widths=.19, patch_artist=True, showfliers=False,
                     medianprops=dict(color='black', lw=1.2, zorder=8), whiskerprops=dict(color='black', lw=.7, zorder=7),
                     capprops=dict(color='black', lw=.7, zorder=7), boxprops=dict(lw=.7, zorder=7))
    for b in bp['boxes']: b.set_facecolor('white'); b.set_alpha(.92); b.set_edgecolor('black'); b.set_zorder(7)
    for el in ('whiskers', 'caps', 'medians'):
        for a in bp[el]: a.set_zorder(8)
    axv.axhline(0, color=RED, lw=0.7, ls=(0, (4, 3.0)), zorder=1)
    axv.set_xticks(range(6)); axv.set_xticklabels([l for _, l, _, _ in NORM], fontsize=XTICK_A, fontweight='bold')
    axv.set_ylabel('$\\bf{Memory\\ performance\\ deviation}$ (SD)', fontsize=7.4, labelpad=3)
    axv.set_ylim(-4.3, 2.9); axv.set_yticks([-4, -2, 0, 2]); axv.tick_params(axis='y', labelsize=6.8)
    axv.set_xlim(-0.62, 5.62)
    bb = axv.get_position()
    fig.text(bb.x0 - 0.062, bb.y1 + 0.006, 'A', fontsize=12.5, fontweight='bold', va='bottom', ha='left')
    for k, (col, title, xlab, (lo, hi), ref) in enumerate(MEAS_RAW):
        r_, c_ = divmod(k, 3)
        ax = fig.add_axes([(GL + c_ * 2.22) / FW1, (1.04 + (1 - r_) * ROWP) / FH1, 1.56 / FW1, 1.40 / FH1])
        g = rows([col, 'ipv_bi_count']); v = g[col].to_numpy(float); n_ = g.ipv_bi_count.to_numpy()
        hw = (hi - lo) / 2.0; ticks = [int(round(ref + kk * hw / 2)) for kk in (-2, -1, 0, 1, 2)]
        sev, non = v[n_ >= 10], v[n_ == 0]
        bins = np.linspace(lo, hi, 13); binw = bins[1] - bins[0]; xk = np.linspace(lo, hi, 300)
        ax.fill_between(xk, kde(non, xk) * 100 * binw, color=GREY_F, alpha=.85, linewidth=0, zorder=2)
        ax.hist(sev, bins=bins, weights=np.full(len(sev), 100 / len(sev)), facecolor=RB_F[k], edgecolor=RB_E[k],
                linewidth=1.2, alpha=0.80, zorder=4)
        ax.axvline(ref, color=RED, lw=1.3, ls=(0, (4, 2.5)), zorder=6)
        ax.set_xlim(lo, hi); ax.set_ylim(0, 44); ax.set_xticks(ticks); ax.tick_params(labelsize=6.6)
        if c_ == 0: ax.set_ylabel('% of group', fontsize=7.8, fontweight='bold')
        ax.set_xlabel(xlab, fontsize=7.6, fontweight='bold')
        ax.set_title(title, fontsize=TITLE_FS, fontweight='bold', loc='left', x=TITLE_X, pad=TITLE_PAD)
        if k == 0:
            gb = ax.get_position()
            fig.text(gb.x0 - 0.062, gb.y1 + B_DY, 'B', fontsize=12.5, fontweight='bold', va='bottom', ha='left')
    _key(fig)
    return fig

TITLE_FS, TITLE_X, TITLE_PAD, B_DY = 8.4, -0.125, 9, 0.012
if __name__ == '__main__':
    f = fig1()
    f.savefig(OUT + '.png', bbox_inches='tight', pad_inches=0.02)
    f.savefig(OUT + '.pdf', bbox_inches='tight', pad_inches=0.02); plt.close(f)
    print('  -> figure written to figures/figure1.png and figures/figure1.pdf | font:', FAM)
