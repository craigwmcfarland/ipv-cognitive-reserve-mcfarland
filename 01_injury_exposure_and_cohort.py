"""
---
title: "Partner-inflicted brain injury exposure and participant characteristics"
output: tables/table1_participant_characteristics.csv
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


header('01  PARTNER-INFLICTED BRAIN INJURY EXPOSURE')
cnt = d['ipv_bi_count'].dropna(); inj = cnt[cnt > 0]
out('Participants', len(cnt))
out('Cumulative partner-inflicted BI count',
    f'median {cnt.median():.0f} (IQR {cnt.quantile(.25):.0f}–{cnt.quantile(.75):.0f}), range {cnt.min():.0f}–{cnt.max():.0f}, '
    f'adjusted Fisher–Pearson skewness G1 = {stats.skew(cnt, bias=False):.2f}')
out('Skewness after winsorizing at 25 and log(1 + count)', f'{stats.skew(np.log1p(np.minimum(cnt, 25)), bias=False):.2f}')
out('Any partner-inflicted BI', f'{len(inj)} ({100 * len(inj) / len(cnt):.0f}%); among them, median {inj.median():.0f} '
    f'(IQR {inj.quantile(.25):.0f}–{inj.quantile(.75):.0f})')
out('Five or more BIs', f'{(cnt >= 5).sum()} ({100 * (cnt >= 5).mean():.0f}%)')
out('Ten or more BIs', f'{(cnt >= 10).sum()} ({100 * (cnt >= 10).mean():.0f}%), of whom '
    f'{len(sample([Y, "ipv_bi_count"], where=lambda q: q.ipv_bi_count >= 10))} completed delayed recall')
out('Exposure bands (0 / 1–9 / ≥10 BIs)', ' / '.join(str(int((d.band == k).sum())) for k in (0, 1, 2)))
q = sample(['years_since_last_ipv_bi', 'ipv_bi_count'], where=lambda q: q.ipv_bi_count > 0)
out('Years since last partner-inflicted BI, injured participants', f'median {q.years_since_last_ipv_bi.median():.1f}')


header('TABLE 1')
T1 = [('Age, years', 'age'), ('Education, years', 'education_years'), ('Post-traumatic stress (PCL-5)', 'ptsd_pcl5'),
      ('Depression (PHQ-9)', 'depression_phq9'), ('Alcohol use (AUDIT)', 'alcohol_audit'),
      ('Childhood maltreatment (CTQ)', 'childhood_maltreatment_ctq'), ('Abuse severity, past year (CAS)', 'abuse_severity_cas')]
fm = lambda m, s: f'{m:.1f} ({s:.1f})'
pk = lambda p: '<.001' if p < .001 else f'{p:.3f}'
kw = lambda groups, v: pk(stats.kruskal(*[g[v] for g in groups]).pvalue)
T1R = []
q = d.dropna(subset=['band']); grp = [q] + [q[q.band == k] for k in (0, 1, 2)]
T1R.append(['n'] + [str(len(g)) for g in grp] + [''])
for lab, v in T1[:2]:
    qq = d.dropna(subset=[v, 'band']); gg = [qq] + [qq[qq.band == k] for k in (0, 1, 2)]
    T1R.append([lab] + [fm(*gm(g[v])) for g in gg] + [kw(gg[1:], v)])
for v, lab in (('non_white', 'Race other than White, n (%)'), ('hispanic', 'Hispanic or Latina, n (%)')):
    qq = d.dropna(subset=[v, 'band']); gg = [qq] + [qq[qq.band == k] for k in (0, 1, 2)]
    T1R.append([lab] + [f'{int(g[v].sum())} ({100 * g[v].mean():.0f})' for g in gg] + [kw(gg[1:], v)])
qq = d.dropna(subset=[RES, 'band']); gg = [qq] + [qq[qq.band == k] for k in (0, 1, 2)]
T1R.append(['Cognitive reserve (ToPF standard score), mean (SD), range']
           + [f'{g[RES].mean():.1f} ({g[RES].std(ddof=1):.1f}), {g[RES].min():.0f}–{g[RES].max():.0f}' for g in gg] + [kw(gg[1:], RES)])
T1R.append(['Partner-inflicted brain injuries, median (IQR)']
           + [f'{g.ipv_bi_count.median():.0f} ({g.ipv_bi_count.quantile(.25):.0f}–{g.ipv_bi_count.quantile(.75):.0f})' for g in grp]
           + [kw(grp[1:], 'ipv_bi_count')])
qq = d.dropna(subset=['years_since_last_ipv_bi', 'band']); qq = qq[qq.band > 0]
T1R.append(['Years since last partner-inflicted injury', fm(*gm(qq.years_since_last_ipv_bi)), '—']
           + [fm(*gm(qq[qq.band == k].years_since_last_ipv_bi)) for k in (1, 2)]
           + [kw([qq[qq.band == k] for k in (1, 2)], 'years_since_last_ipv_bi')])
for lab, v in T1[2:]:
    qq = d.dropna(subset=[v, 'band']); gg = [qq] + [qq[qq.band == k] for k in (0, 1, 2)]
    T1R.append([lab] + [fm(*gm(g[v])) for g in gg] + [kw(gg[1:], v)])
for r_ in T1R:
    out(r_[0], ' | '.join(r_[1:]))
save_table('table1_participant_characteristics', ['Characteristic', 'All', '0', '1–9', '≥10', 'P'], T1R)
