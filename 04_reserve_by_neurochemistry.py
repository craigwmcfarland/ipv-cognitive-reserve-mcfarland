"""
---
title: "Cognitive reserve by brain metabolites on memory"
output: tables/supp_table7_metabolite_grid.csv, tables/supp_table8_sensitivity_reserve_by_tnaa.csv
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
_pre = pd.read_csv(os.path.join(DATA, 'IPV_CognitiveReserve_DATA_original_MRS_inclusion.csv'))
PRE_QC = {v: int(_pre[f'naa_naag_crpcr_{v}'].notna().sum()) for v in ('pcg', 'dlpfc', 'coro')}
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


header('04  COGNITIVE RESERVE, NEUROCHEMISTRY AND MEMORY')
VOX = {'pcg': 'PCG', 'dlpfc': 'DLPFC', 'coro': 'Corona radiata'}
MET = {'naa_naag': 'tNAA/tCr', 'glu_gln': 'Glx', 'mi': 'mI', 'pch_gpc': 'Cho'}
PCG = 'naa_naag_crpcr_pcg'
out('Spectra before quality control (PCG / DLPFC / corona radiata)', ' / '.join(str(PRE_QC[v]) for v in VOX))
out('Spectra after quality control', ' / '.join(str(d[f'naa_naag_crpcr_{v}'].notna().sum()) for v in VOX))
mrs_in = d[PCG].notna()
print('  Women with and without spectroscopy (Mann–Whitney P)')
for v, lab in (('age', 'age'), ('education_years', 'education'), (RES, 'cognitive reserve'), ('log_ipv_bi_capped', 'injury burden'),
               (Y, 'delayed recall'), ('ptsd_pcl5', 'PTSD symptoms')):
    out(f'    {lab}', fp(stats.mannwhitneyu(d.loc[mrs_in, v].dropna(), d.loc[~mrs_in, v].dropna()).pvalue))


grid = {}
for vx in VOX:
    for mt in MET:
        q, M = moderation(f'{mt}_crpcr_{vx}'); grid[(vx, mt)] = (len(q), M.est(3))
fam1 = [(v, 'naa_naag') for v in VOX]; fam2 = [k for k in grid if k not in fam1]
qf = dict(zip(fam1, stats.false_discovery_control([grid[k][1][3] for k in fam1])))
qf.update(zip(fam2, stats.false_discovery_control([grid[k][1][3] for k in fam2])))
print('\n  Reserve x metabolite interaction on delayed recall (Supplementary Table 7)')
for k, (n, t) in grid.items():
    row('ST7', [VOX[k[0]], MET[k[1]]] + est_cells(n, t) + [P(qf[k])])
    out(f'  {VOX[k[0]]} {MET[k[1]]}', B(t) + f', q = {fp(qf[k])}, n = {n}')
print('  tNAA/tCr by voxel, analysis samples')
for v in VOX:
    x_ = moderation(f'naa_naag_crpcr_{v}')[0][f'naa_naag_crpcr_{v}']
    out(f'    {VOX[v]}', f'mean {x_.mean():.3f}, SD {x_.std(ddof=1):.3f}, range {x_.min():.2f}–{x_.max():.2f}, '
        f'-1/+1 SD {x_.mean() - x_.std(ddof=1):.2f}/{x_.mean() + x_.std(ddof=1):.2f}, median {x_.median():.2f} '
        f'({(x_ <= x_.median()).sum()} at or below, {(x_ > x_.median()).sum()} above), Shapiro–Wilk P = {fp(stats.shapiro(x_).pvalue)}')


qN, MN = moderation(PCG)
print('\n  PCG tNAA/tCr model')
out('  Reserve slope at -1 SD tNAA/tCr', B(MN.combo(1, 3, -1)))
out('  Reserve slope at +1 SD tNAA/tCr', B(MN.combo(1, 3, 1)))
out('  tNAA/tCr slope at -1 SD reserve', B(MN.combo(2, 3, -1)))
out('  tNAA/tCr slope at +1 SD reserve', B(MN.combo(2, 3, 1)))

yN, tN, mN = z(qN[Y]), z(qN[RES]), z(qN[PCG]); CN = [z(qN[v]) for v in MRSCOV]
rng = np.random.default_rng(SEED); boot = []
for _ in range(10000):
    i = rng.integers(0, len(yN), len(yN))
    boot.append(OLS(yN[i], [tN[i], mN[i], tN[i] * mN[i]] + [c_[i] for c_ in CN]).b[3])
boot = np.array(boot); th = MN.b[3]
z0 = stats.norm.ppf(np.mean(boot < th))
jk = np.array([OLS(np.delete(yN, k), [np.delete(tN, k), np.delete(mN, k), np.delete(tN * mN, k)] + [np.delete(c_, k) for c_ in CN]).b[3] for k in range(len(yN))])
acc = np.sum((jk.mean() - jk) ** 3) / (6 * np.sum((jk.mean() - jk) ** 2) ** 1.5)
adj = lambda al: stats.norm.cdf(z0 + (z0 + stats.norm.ppf(al)) / (1 - acc * (z0 + stats.norm.ppf(al))))
bci = np.percentile(boot, [100 * adj(.025), 100 * adj(.975)])
out('  BCa bootstrap 95% CI (10,000 resamples)', f'{bci[0]:.2f} to {bci[1]:.2f}; {100 * np.mean(boot < 0):.1f}% of estimates below zero')
LOON = [moderation(PCG, drop=(i,))[1].est(3) for i in qN.ID]
out('  Leaving out each participant in turn', f'β = {min(x[0] for x in LOON):.2f} to {max(x[0] for x in LOON):.2f}, largest P = {fp(max(x[3] for x in LOON))}')


def cv_r2(y, cols):
    y = np.asarray(y, float); X = np.column_stack([np.ones(len(y))] + list(cols)); e = []
    for i in range(len(y)):
        k_ = np.arange(len(y)) != i
        e.append(y[i] - X[i] @ np.linalg.lstsq(X[k_], y[k_], rcond=None)[0])
    e = np.array(e); return 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()


out('  Leave-one-out cross-validated R², without -> with the interaction', f'{cv_r2(yN, [tN, mN] + CN):.3f} -> {cv_r2(yN, [tN, mN, tN * mN] + CN):.3f}')


print('\n  Sensitivity analyses for the reserve-by-PCG tNAA/tCr interaction (Supplementary Table 8)')
row('ST8', ['Primary model'] + est_cells(len(qN), MN.est(3)))
row('ST8', ['Bias-corrected and accelerated bootstrap', I(len(qN)), '—', D(2)(bci[0]), D(2)(bci[1]), '—'])
row('ST8', ['Leave-one-out, range across omitted participants', I(len(qN) - 1),
            f'{D(2)(min(x[0] for x in LOON))} to {D(2)(max(x[0] for x in LOON))}', '—', '—', f'max {P(max(x[3] for x in LOON))}'])
for lab, ex in (('Adjusted for body mass index', ['BMI']), ('Adjusted for any medication use', ['any_medication']),
                ('Adjusted for psychotropic medication use', ['psychotropic']),
                ('Adjusted for body mass index and both medication variables', ['BMI', 'any_medication', 'psychotropic']),
                ('Additionally adjusted for depression and alcohol use', ['BMI', 'any_medication', 'psychotropic', 'depression_phq9', 'alcohol_audit'])):
    q, M = moderation(PCG, cov=MRSCOV + ex)
    row('ST8', [lab] + est_cells(len(q), M.est(3))); out(f'  {lab}', B(M.est(3)))
q, M = moderation(PCG, extra_terms=lambda q, t, m: [t * z(q.age)])
row('ST8', ['Reserve-by-age term added'] + est_cells(len(q), M.est(3))); out('  Reserve-by-age term added', B(M.est(3)))


print('\n  Joint and component models')
q2 = sample([Y, RES, PCG, 'ipv_bi_count'] + MRSCOV)
t2, m2 = z(q2[RES]), z(q2[PCG]); h2 = (q2.ipv_bi_count >= 10).astype(float).values
M2 = OLS(z(q2[Y]), [t2, m2, t2 * m2, h2, t2 * h2] + [z(q2[v]) for v in MRSCOV])
out('  Both moderators: reserve x PCG tNAA/tCr', B(M2.est(3)) + f', n = {len(q2)}')
out('  Both moderators: reserve x ten or more BIs', B(M2.est(5)))
M3 = OLS(z(q2[Y]), [t2, m2, h2, t2 * m2, t2 * h2, m2 * h2, t2 * m2 * h2] + [z(q2[v]) for v in MRSCOV])
out('  Reserve x PCG tNAA/tCr x ten or more BIs', B(M3.est(7)))
out('  Reserve x tNAA (alone)', B(moderation('naa_naag_pcg')[1].est(3)))
out('  Reserve x tCr (alone)', B(moderation('cr_pcr_pcg')[1].est(3)))
qc2 = sample([Y, RES, 'naa_naag_pcg', 'cr_pcr_pcg'] + MRSCOV)
tc2, na2, cr2 = z(qc2[RES]), z(qc2.naa_naag_pcg), z(qc2.cr_pcr_pcg)
Mc2 = OLS(z(qc2[Y]), [tc2, na2, cr2, tc2 * na2, tc2 * cr2] + [z(qc2[v]) for v in MRSCOV])
out('  Both components: reserve x tNAA', B(Mc2.est(4)))
out('  Both components: reserve x tCr', B(Mc2.est(5)))
out('  Reserve x PCG tNAA/tCr on the CVLT-3 episodic composite', B(moderation(PCG, y='cvlt_epi')[1].est(3)))
out('  Reserve x PCG tNAA/tCr on CNSVS visual memory', B(moderation(PCG, y='visual_memory')[1].est(3)))
qD, Md = moderation('naa_naag_crpcr_dlpfc')
LOOD = [moderation('naa_naag_crpcr_dlpfc', drop=(i,))[1].est(3) for i in qD.ID]
out('  DLPFC: reserve slope at -1 SD tNAA/tCr', B(Md.combo(1, 3, -1)))
out('  DLPFC: leaving out each participant in turn', f'β = {min(x[0] for x in LOOD):.2f} to {max(x[0] for x in LOOD):.2f}, largest P = {fp(max(x[3] for x in LOOD))}')
_, Mc = moderation('pch_gpc_crpcr_coro')
out('  Corona radiata Cho: reserve slope at +1 SD Cho', B(Mc.combo(1, 3, 1)))
print('  Metabolites vs delayed recall and vs number of BIs (Spearman)')
for col, lab in ((PCG, 'PCG tNAA/tCr'), ('naa_naag_crpcr_dlpfc', 'DLPFC tNAA/tCr'), ('pch_gpc_crpcr_coro', 'Corona radiata Cho')):
    for other, ol in ((Y, 'delayed recall'), ('ipv_bi_count', 'number of BIs')):
        q = sample([col, other]); out(f'    {lab} vs {ol}', RHO(*spear(q[col], q[other]), len(q)))


qm_ = sample([Y, PCG] + MRSCOV)
e, m, yy = z(qm_.log_ipv_bi_capped), z(qm_[PCG]), z(qm_[Y]); C3 = [z(qm_[v]) for v in ('age', 'education_years', 'ptsd_pcl5')]
Ma, Mb = OLS(m, [e] + C3), OLS(yy, [e, m] + C3)
rng = np.random.default_rng(SEED); ab = []
for _ in range(10000):
    i = rng.integers(0, len(yy), len(yy))
    ab.append(OLS(m[i], [e[i]] + [c_[i] for c_ in C3]).b[1] * OLS(yy[i], [e[i], m[i]] + [c_[i] for c_ in C3]).b[2])
ci = np.percentile(ab, [2.5, 97.5])
print('\n  Mediation: injury burden -> PCG tNAA/tCr -> delayed recall')
out('  a path', B(Ma.est(1))); out('  b path', B(Mb.est(2))); out('  Direct effect', B(Mb.est(1)))
out('  Indirect effect a x b', f'{Ma.b[1] * Mb.b[2]:.4f}, 95% bootstrap CI {ci[0]:.2f} to {ci[1]:.2f}')

header('SUPPLEMENTARY TABLES 7 AND 8')
EST = ['n', 'β', '95% CI lower', '95% CI upper', 'P']
save_table('supp_table7_metabolite_grid', ['Voxel', 'Metabolite'] + EST + ['q (family)'], TABLE_ROWS['ST7'])
save_table('supp_table8_sensitivity_reserve_by_tnaa', ['Specification'] + EST, TABLE_ROWS['ST8'])
