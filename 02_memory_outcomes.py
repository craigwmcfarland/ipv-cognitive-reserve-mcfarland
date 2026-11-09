"""
---
title: "Memory outcomes and variance explained by injury burden and cognitive reserve"
output: tables/supp_table1_memory_performance.csv, tables/supp_table2_variance_explained.csv
---
"""
import math, os, csv
import numpy as np, pandas as pd
from scipy import stats, optimize

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
TABLES = os.path.join(HERE, 'tables')
SEED = 20260827
Y, RES = 'np_ldfr_scaled', 'np_topf_ss'
MEMCOV = ['education_years', 'ptsd_pcl5']
MRSCOV = ['log_ipv_bi_capped', 'age', 'education_years', 'ptsd_pcl5']
MEM = ['composite_memory', 'verbal_memory', 'visual_memory', 'working_memory']
NON = ['cognitive_flexibility', 'complex_attention', 'executive_function', 'motor_speed',
       'processing_speed', 'psychomotor_speed', 'reaction_time', 'reasoning',
       'simple_attention', 'sustained_attention', 'social_acuity']
FLAGGED = ['IPV006', 'IPV071', 'IPV197']


d = pd.read_csv(os.path.join(DATA, 'IPV_CognitiveReserve_DATA.csv'))
d = d.merge(pd.read_csv(os.path.join(DATA, 'IPV_BMI_medication.csv')), on='ID', how='left')
d = d.merge(pd.read_csv(os.path.join(DATA, 'IPV_MRS_PCG_components.csv')), on='ID', how='left')
d['cvlt_epi'] = d[[Y, 'cvlt_learn']].mean(axis=1, skipna=False)


CNSVS = ['composite_memory', 'verbal_memory', 'visual_memory', 'working_memory', 'cognitive_flexibility',
         'complex_attention', 'executive_function', 'motor_speed', 'processing_speed', 'psychomotor_speed',
         'reaction_time', 'reasoning', 'simple_attention', 'sustained_attention', 'social_acuity']
OUTLIER_SD = 4
REMOVED = {}
for c_ in CNSVS:
    out = (d[c_] - 100).abs() > OUTLIER_SD * 15
    REMOVED[c_] = d.loc[out, 'ID'].tolist()
    d.loc[out, c_] = np.nan
c = d['ipv_bi_count']
d['band'] = np.where(c == 0, 0, np.where(c < 10, 1, 2)).astype(float)
d.loc[c.isna(), 'band'] = np.nan
d['hi10'] = (c >= 10).astype(float); d.loc[c.isna(), 'hi10'] = np.nan
d['log_uncapped'] = np.log1p(c)
d['count_capped'] = np.minimum(c, 25)
d['log_blunt'] = np.log1p(np.minimum(d['blunt_force_bi_count'], 25))
d['log_strang'] = np.log1p(np.minimum(d['strangulation_bi_count'], 25))
d['strang_ever'] = (d['strangulation_bi_count'] > 0).astype(float)
d.loc[d['strangulation_bi_count'].isna(), 'strang_ever'] = np.nan


def z(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / v.std(ddof=1)


class OLS:
    def __init__(self, y, cols):
        X = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in cols])
        y = np.asarray(y, float)
        self.n, self.k = X.shape
        XtXi = np.linalg.pinv(X.T @ X)
        self.b = XtXi @ X.T @ y
        self.resid = y - X @ self.b
        h = np.einsum('ij,jk,ik->i', X, XtXi, X)
        self.cov = XtXi @ (X.T * ((self.resid / (1 - h)) ** 2)) @ X @ XtXi
        self.se = np.sqrt(np.diag(self.cov))
        self.df = self.n - self.k
        self.tc = stats.t.ppf(.975, self.df)

    def est(self, i):
        b, s = self.b[i], self.se[i]
        return b, b - self.tc * s, b + self.tc * s, 2 * stats.t.sf(abs(b / s), self.df)

    def lin(self, w):
        w = np.asarray(w, float); b = w @ self.b; s = math.sqrt(w @ self.cov @ w)
        return b, b - self.tc * s, b + self.tc * s, 2 * stats.t.sf(abs(b / s), self.df)

    def combo(self, i, j, sign=1):
        w = np.zeros(self.k); w[i] = 1; w[j] = sign
        return self.lin(w)


def sample(cols, drop=(), where=None):
    q = d.dropna(subset=list(dict.fromkeys(cols)))
    q = q[~q.ID.isin(drop)]
    return q[where(q)] if where is not None else q


def strata_model(y=Y, cov=MEMCOV, extra=(), drop=(), k=10):
    q = sample([y, RES, 'ipv_bi_count'] + list(cov) + list(extra), drop)
    t = z(q[RES]); mid = ((q.ipv_bi_count > 0) & (q.ipv_bi_count < k)).astype(float).values
    hi = (q.ipv_bi_count >= k).astype(float).values
    return q, OLS(z(q[y]), [t, mid, hi, t * mid, t * hi] + [z(q[v]) for v in list(cov) + list(extra)])


def binary_model(y=Y, cov=MEMCOV, extra=(), drop=(), k=10):
    q = sample([y, RES, 'ipv_bi_count'] + list(cov) + list(extra), drop)
    t = z(q[RES]); hi = (q.ipv_bi_count >= k).astype(float).values
    return q, OLS(z(q[y]), [t, hi, t * hi] + [z(q[v]) for v in list(cov) + list(extra)])


def moderation(mod, y=Y, cov=MRSCOV, extra_terms=None, drop=()):
    q = sample([y, RES, mod] + list(cov), drop)
    t, m = z(q[RES]), z(q[mod])
    cols = [t, m, t * m] + [z(q[v]) for v in cov]
    if extra_terms:
        cols += extra_terms(q, t, m)
    return q, OLS(z(q[y]), cols)


def spear(a, b):
    r = stats.spearmanr(a, b); return r.statistic, r.pvalue


def pct(score, mu=10, sd=3):
    return 100 * stats.norm.cdf((score - mu) / sd)


def ri_lmm(y, X, g):
    y = np.asarray(y, float); X = np.asarray(X, float); g = np.asarray(g)
    groups = [np.where(g == u)[0] for u in np.unique(g)]
    n, p = X.shape

    def parts(lam):
        A = np.zeros((p, p)); v = np.zeros(p); logdet = 0.0; bl = []
        for idx in groups:
            m_ = len(idx); cc = lam / (1 + m_ * lam); sx = X[idx].sum(0)
            A += X[idx].T @ X[idx] - cc * np.outer(sx, sx)
            v += X[idx].T @ y[idx] - cc * sx * y[idx].sum()
            logdet += math.log(1 + m_ * lam); bl.append((idx, cc))
        b = np.linalg.solve(A, v); r = y - X @ b
        rss = sum(r[i] @ r[i] - cc * r[i].sum() ** 2 for i, cc in bl)
        return b, rss, logdet, A

    def nll(ll):
        b, rss, logdet, A = parts(math.exp(ll))
        return 0.5 * ((n - p) * math.log(rss / (n - p)) + logdet + np.linalg.slogdet(A)[1])
    lam = math.exp(optimize.minimize_scalar(nll, bounds=(-12, 6), method='bounded').x)
    b, rss, logdet, A = parts(lam)
    cov = rss / (n - p) * np.linalg.inv(A)
    se = np.sqrt(np.diag(cov))
    return b, se, len(groups) - 1


TABLE_ROWS = {}
gm = lambda x: [x.mean(), x.std(ddof=1)]


def fp(p):
    return f'{p:.1e}' if p < 1e-4 else f'{p:.4f}' if p < .01 else f'{p:.3f}'


def B(t):
    b, lo, hi, p = t
    return f'β = {b:.3f}, 95% CI [{lo:.2f}, {hi:.2f}], P = {fp(p)}'


def RHO(r, p, n=None):
    return f'ρ = {r:.3f}, P = {fp(p)}' + (f', n = {n}' if n is not None else '')


def out(label, text):
    print(f'  {label}: {text}')


def header(t):
    print('\n' + '=' * 100 + f'\n{t}\n' + '=' * 100)


D = lambda k: (lambda v: (f'{float(v):.{k}f}'.lstrip('-') if round(float(v), k) == 0 else f'{float(v):.{k}f}'))
I = lambda v: str(int(round(float(v))))
P = lambda p: '<.001' if p < .001 else f'{p:.3f}'.replace('0.', '.', 1)


def row(table, cells):
    TABLE_ROWS.setdefault(table, []).append(cells)


def est_cells(n, t):
    return [I(n), D(3)(t[0]), D(2)(t[1]), D(2)(t[2]), P(t[3])]


def save_table(name, cols, rows):
    os.makedirs(TABLES, exist_ok=True)
    with open(os.path.join(TABLES, f'{name}.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh); w.writerow(cols); w.writerows(rows)
    print(f'  -> table written to tables/{name}.csv')


header('02  MEMORY OUTCOMES')
nrem = sum(len(v) for v in REMOVED.values())
out('CNS Vital Signs standard scores beyond 4 SD removed', f'{nrem} of {d[CNSVS].shape[0] * len(CNSVS)} '
    f'({sum(len(REMOVED[c_]) for c_ in MEM)} in memory domains)')

MEAS = [(Y, 'CVLT-3 long-delay free recall', 10, 3), ('cvlt_learn', 'CVLT-3 verbal learning', 10, 3),
        ('np_dr_index', 'CVLT-3 Delayed Memory Index', 100, 15), ('composite_memory', 'CNSVS memory composite', 100, 15),
        ('verbal_memory', 'CNSVS verbal memory', 100, 15), ('visual_memory', 'CNSVS visual memory', 100, 15),
        ('working_memory', 'CNSVS working memory', 100, 15)]
print('\n  Range of scores as percentiles of the normative population')
for col, lab, mu, sd in MEAS:
    x = d[col].dropna()
    out(f'  {lab}', f'{pct(x.min(), mu, sd):.2f} to {pct(x.max(), mu, sd):.2f}')
hb = sample([Y, 'ipv_bi_count'], where=lambda q: q.ipv_bi_count >= 10)[Y]
out('Delayed recall, ten or more BIs', f'{pct(hb.min()):.1f} to {pct(hb.max()):.1f} percentile, SD {hb.std(ddof=1):.2f} (normative SD 3)')
ten = sample([Y, 'ipv_bi_count'], where=lambda q: q.ipv_bi_count == 10)[Y].sort_values()
out('Delayed recall, participants reporting exactly ten BIs (percentiles)', ', '.join(f'{pct(v):.0f}' for v in ten))

q = sample([Y, 'log_ipv_bi_capped', 'education_years'])
M = OLS(z(q[Y]), [z(q.log_ipv_bi_capped), z(q.education_years)])
out('Injury burden -> delayed recall, adjusted for education', B(M.est(1)) + f', n = {len(q)}')


NORM = [(Y, 'CVLT delayed recall', 10, 3), ('cvlt_learn', 'CVLT verbal learning', 10, 3),
        ('np_dr_index', 'CVLT-3 Delayed Memory Index', 100, 15), ('composite_memory', 'CNSVS memory composite', 100, 15),
        ('verbal_memory', 'CNSVS verbal memory', 100, 15), ('visual_memory', 'CNSVS visual memory', 100, 15)]
print('\n  Memory performance: cohort, no injury and ten or more BIs (Supplementary Table 1)')
for v, lab, mu, sd in NORM:
    x = d[v].dropna(); x0 = d[d.band == 0][v].dropna(); x2 = d[d.band == 2][v].dropna()
    p = stats.ttest_1samp(x, mu).pvalue
    cells = [lab, I(len(x)), D(1)(x.mean()), D(1)(x.std(ddof=1)), D(2)((x.mean() - mu) / sd), I(len(x0)), D(1)(x0.mean()),
             D(1)(x0.std(ddof=1)), I(len(x2)), D(1)(x2.mean()), D(1)(x2.std(ddof=1)), P(p)]
    row('ST1', cells); out(f'  {lab}', ', '.join(cells[1:]))


ST2 = {}
meas = [(Y, 'CVLT long-delay free recall'), ('cvlt_learn', 'CVLT trials 1 to 5 learning'),
        ('cvlt_epi', 'CVLT episodic composite'), ('np_dr_index', 'CVLT-3 Delayed Memory Index'),
        ('composite_memory', 'CNSVS composite memory'), ('verbal_memory', 'CNSVS verbal memory'),
        ('visual_memory', 'CNSVS visual memory'), ('working_memory', 'CNSVS working memory')]
bands = [('Whole cohort', lambda q: q.band >= 0), ('No injury', lambda q: q.band == 0),
         ('1 to 9 injuries', lambda q: q.band == 1), ('10 or more injuries', lambda q: q.band == 2)]
print('\n  Variance explained (Supplementary Table 2): n; injury rho² (%), P, r² (%); reserve rho² (%), P, r² (%)')
for col, lab in meas:
    for bl, sel in bands:
        e = 'log_ipv_bi_capped' if bl == 'Whole cohort' else 'ipv_bi_count'
        r = {}
        if bl != 'No injury':
            g = sample([col, 'ipv_bi_count', 'log_ipv_bi_capped'], where=sel)
            rs, ps = spear(g[e], g[col]); rp = stats.pearsonr(g[e], g[col])[0]
            r['inj'] = (len(g), 100 * rs ** 2, ps, 100 * rp ** 2)
        g = sample([col, RES, 'ipv_bi_count'], where=sel)
        rs, ps = spear(g[RES], g[col]); rp = stats.pearsonr(g[RES], g[col])[0]
        r['res'] = (len(g), 100 * rs ** 2, ps, 100 * rp ** 2)
        ST2[(lab, bl)] = r
        inj_cells = [D(1)(r['inj'][1]), P(r['inj'][2]), D(1)(r['inj'][3])] if 'inj' in r else ['—', '—', '—']
        cells = [lab, bl, I(r['res'][0])] + inj_cells + [D(1)(r['res'][1]), P(r['res'][2]), D(1)(r['res'][3])]
        row('ST2', cells); out(f'  {lab}, {bl}', ', '.join(cells[2:]))
inj10 = [ST2[(lab, '10 or more injuries')]['inj'] for _, lab in meas]
res10 = [ST2[(lab, '10 or more injuries')]['res'] for _, lab in meas]
rec10 = ST2[('CVLT long-delay free recall', '10 or more injuries')]
print()
out('Largest whole-cohort variance explained by injury burden (Pearson r², %)',
    f'{max(ST2[(lab, "Whole cohort")]["inj"][3] for _, lab in meas):.1f}')
out('Ten or more BIs, injury burden across measures',
    f'rho² {min(x[1] for x in inj10):.1f}% to {max(x[1] for x in inj10):.1f}%, Pearson r² {min(x[3] for x in inj10):.1f}% to '
    f'{max(x[3] for x in inj10):.1f}%, smallest P = {fp(min(x[2] for x in inj10))}')
out('Ten or more BIs, cognitive reserve across measures (Pearson r², %)', f'{min(x[3] for x in res10):.1f} to {max(x[3] for x in res10):.1f}')
out('Ten or more BIs, delayed recall (Pearson r², %)', f'injury burden {rec10["inj"][3]:.1f}, cognitive reserve {rec10["res"][3]:.1f} '
    f'(n = {rec10["res"][0]})')

header('SUPPLEMENTARY TABLES 1 AND 2')
save_table('supp_table1_memory_performance',
           ['Measure', 'n', 'Cohort mean', 'SD', 'Deviation (normative SD)', 'n, no injury', 'No injury mean', 'SD',
            'n, ten or more', 'Ten or more mean', 'SD', 'P'], TABLE_ROWS['ST1'])
save_table('supp_table2_variance_explained', ['Memory measure', 'Exposure band', 'n', 'Injury rho² (%)', 'P', 'Injury r² (%)',
                                              'Reserve rho² (%)', 'P', 'Reserve r² (%)'], TABLE_ROWS['ST2'])
