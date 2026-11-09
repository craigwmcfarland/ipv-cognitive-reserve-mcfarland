"""
---
title: "Cognitive reserve by injury burden on memory"
output: tables/supp_table3_sensitivity_reserve_by_injury.csv, tables/supp_table4_exposure_thresholds.csv, tables/supp_table5_injury_mechanism.csv, tables/supp_table6_cnsvs_domains.csv
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


header('03  COGNITIVE RESERVE, INJURY BURDEN AND MEMORY')
cnt = d['ipv_bi_count'].dropna()

print('\n  Cognitive reserve -> memory, adjusted for injury burden, education and PTSD symptoms')
for col, lab in ((Y, 'Delayed recall'), ('cvlt_learn', 'CVLT-3 verbal learning'), ('cvlt_epi', 'CVLT-3 episodic composite'),
                 ('np_dr_index', 'CVLT-3 Delayed Memory Index'), ('composite_memory', 'CNSVS memory composite'),
                 ('verbal_memory', 'CNSVS verbal memory'), ('visual_memory', 'CNSVS visual memory'),
                 ('working_memory', 'CNSVS working memory')):
    cov = ['log_ipv_bi_capped'] + MEMCOV
    q = sample([col, RES] + cov)
    out(f'  {lab}', B(OLS(z(q[col]), [z(q[RES])] + [z(q[v]) for v in cov]).est(1)) + f', n = {len(q)}')
print('  Cognitive reserve -> delayed recall under other adjustment sets')
for cov, lab in ((['log_ipv_bi_capped'], 'injury burden'), (['log_ipv_bi_capped', 'education_years'], 'injury burden and education')):
    q = sample([Y, RES] + cov)
    out(f'  adjusted for {lab}', B(OLS(z(q[Y]), [z(q[RES])] + [z(q[v]) for v in cov]).est(1)))


print('\n  Cognitive reserve and delayed recall within exposure bands (Spearman)')
strat = {}
for bl, sel in (('No partner-inflicted BI', lambda q: q.band == 0), ('One to nine BIs', lambda q: q.band == 1),
                ('Ten or more BIs', lambda q: q.band == 2), ('Fewer than ten BIs', lambda q: q.band < 2)):
    g = sample([Y, RES, 'band'], where=sel); strat[bl] = g
    out(f'  {bl}', RHO(*spear(g[RES], g[Y]), len(g)))
g10 = strat['Ten or more BIs']
loo = [spear(g10.drop(i)[RES], g10.drop(i)[Y]) for i in g10.index]
out('  Ten or more BIs, leaving out each participant in turn',
    f'ρ = {min(x[0] for x in loo):.2f} to {max(x[0] for x in loo):.2f}, largest P = {fp(max(x[1] for x in loo))}')


qP, MP = strata_model()
s10 = MP.combo(1, 5)
print('\n  Reserve-by-injury interaction on delayed recall (three exposure strata; education and PTSD adjusted)')
out('  Reserve x ten or more BIs (vs none)', B(MP.est(5)) + f', n = {len(qP)}')
out('  Reserve slope, no partner-inflicted BI', B(MP.est(1)))
out('  Reserve slope, ten or more BIs', B(s10))
shift = s10[0] * qP[Y].std(ddof=1)
out('  One SD of reserve in the highest band, as normative percentiles',
    f'{s10[0]:.2f} SD of recall = {shift:.2f} scaled-score points; {pct(10 - shift):.0f}th vs {pct(10):.0f}th percentile')
LOOI = [strata_model(drop=(pid,))[1].est(5) for pid in qP.ID]
out('  Leaving out each participant in turn', f'β = {min(x[0] for x in LOOI):.3f} to {max(x[0] for x in LOOI):.3f}; '
    f'P ≥ .05 for {sum(x[3] >= .05 for x in LOOI)} of {len(LOOI)}; largest P = {fp(max(x[3] for x in LOOI))}')

yP, tP = z(qP[Y]), z(qP[RES]); midP = (qP.band == 1).astype(float).values; hiP = (qP.band == 2).astype(float).values
cvP = [z(qP[v]) for v in MEMCOV]; obs = MP.b[5] / MP.se[5]
Xr = np.column_stack([np.ones(len(yP)), tP, midP, hiP, tP * midP] + cvP)
fitr = Xr @ np.linalg.lstsq(Xr, yP, rcond=None)[0]; resr = yP - fitr
rng = np.random.default_rng(SEED); n_ge = 0
for _ in range(10000):
    Mp = OLS(fitr + rng.permutation(resr), [tP, midP, hiP, tP * midP, tP * hiP] + cvP)
    n_ge += abs(Mp.b[5] / Mp.se[5]) >= abs(obs)
P_FL = (n_ge + 1) / 10001
out('  Freedman–Lane permutation test (10,000 permutations)', f'P = {fp(P_FL)}')


def cv_r2(y, cols):
    y = np.asarray(y, float); X = np.column_stack([np.ones(len(y))] + list(cols)); e = []
    for i in range(len(y)):
        k_ = np.arange(len(y)) != i
        e.append(y[i] - X[i] @ np.linalg.lstsq(X[k_], y[k_], rcond=None)[0])
    e = np.array(e); return 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()


out('  Leave-one-out cross-validated R², without -> with the interaction',
    f'{cv_r2(yP, [tP, midP, hiP] + cvP):.3f} -> {cv_r2(yP, [tP, midP, hiP, tP * midP, tP * hiP] + cvP):.3f}')
q12, M12 = strata_model(k=12)
out('  Three strata with the highest band at twelve or more BIs', B(M12.est(5)) + f', {int((q12.ipv_bi_count >= 12).sum())} in the band')
out('  Participants reporting exactly 6, 9 or 11 BIs', ', '.join(f'{k}: {int((cnt == k).sum())}' for k in (6, 9, 11)))
qc = sample([Y, RES, 'ipv_bi_count'] + MEMCOV); tc_ = z(qc[RES])
print('  Injury burden coded continuously')
for lab, e in (('raw count', qc.ipv_bi_count), ('log(1 + count)', np.log1p(qc.ipv_bi_count)),
               ('square root', np.sqrt(qc.ipv_bi_count)), ('rank', stats.rankdata(qc.ipv_bi_count))):
    e = z(e); out(f'    reserve x {lab}', B(OLS(z(qc[Y]), [tc_, e, tc_ * e] + [z(qc[v]) for v in MEMCOV]).est(3)))


print('\n  Reserve x highest band (binary model) at each threshold (Supplementary Table 4)')
for k in range(4, 13):
    q, M = binary_model(k=k); hi = int((q.ipv_bi_count >= k).sum())
    row('ST4', [f'≥{k}', I(hi), I(len(q) - hi)] + est_cells(len(q), M.est(3))[1:])
    out(f'  ≥{k} ({hi} above, {len(q) - hi} below)', B(M.est(3)))


print('\n  Sensitivity analyses for the reserve-by-injury interaction (Supplementary Table 3)')
ST3 = [('Primary model, three exposure strata', {}),
       ('Binary, ten or more against fewer', 'binary'),
       ('Additionally adjusted for injury count, log-transformed and uncapped', {'extra': ['log_uncapped']}),
       ('Additionally adjusted for injury count, untransformed and capped at 25', {'extra': ['count_capped']}),
       ('Adjusted for depression', {'extra': ['depression_phq9']}),
       ('Adjusted for childhood maltreatment', {'extra': ['childhood_maltreatment_ctq']}),
       ('Adjusted for alcohol use', {'extra': ['alcohol_audit']}),
       ('Adjusted for other sources of brain injury', {'extra': ['log_childhood_bi', 'log_nonpartner_adult_bi']}),
       ('Adjusted for past-year abuse severity', {'extra': ['abuse_severity_cas']}),
       ('Education removed from the model', {'cov': ['ptsd_pcl5']}),
       ('Adjusted for body mass index', {'extra': ['BMI']}),
       ('Adjusted for medication use', {'extra': ['any_medication']}),
       ('Adjusted for psychotropic medication use', {'extra': ['psychotropic']})]
for lab, kw in ST3:
    q, M = binary_model() if kw == 'binary' else strata_model(**kw)
    t = M.est(3 if kw == 'binary' else 5)
    row('ST3', [lab] + est_cells(len(q), t)); out(f'  {lab}', B(t) + f', n = {len(q)}')
row('ST3', ['Leave-one-out, range across omitted participants', I(len(qP) - 1),
            f'{D(3)(min(x[0] for x in LOOI))} to {D(3)(max(x[0] for x in LOOI))}', '—', '—', f'max {P(max(x[3] for x in LOOI))}'])


qm = sample([Y, RES, 'log_blunt', 'log_strang', 'ipv_bi_count'] + MEMCOV)
tm_ = z(qm[RES]); bb = z(qm.log_blunt); ss = z(qm.log_strang)
MM = OLS(z(qm[Y]), [tm_, bb, ss, tm_ * bb, tm_ * ss] + [z(qm[v]) for v in MEMCOV])
qs_ = sample([Y, RES, 'strang_ever', 'log_ipv_bi_capped'] + MEMCOV); ts_ = z(qs_[RES]); se_ = qs_.strang_ever.values
Ms = OLS(z(qs_[Y]), [ts_, se_, z(qs_.log_ipv_bi_capped), ts_ * se_] + [z(qs_[v]) for v in MEMCOV])
print('\n  Injury mechanism (Supplementary Table 5)')
injm = qm[qm.ipv_bi_count > 0]; sm = qm[qm.strangulation_bi_count > 0].strangulation_bi_count
out('  Events in the cohort', f'strangulation alone {int(d.strangulation_bi_count.sum())}, blunt-force category {int(d.blunt_force_bi_count.sum())}')
out('  Blunt force vs strangulation counts among injured participants', RHO(*spear(injm.log_blunt, injm.log_strang), len(injm)))
out('  Participants reporting any strangulation', f'{len(sm)}; median count {sm.median():.0f}; ten or more: {(sm >= 10).sum()}')
for lab, t, n in (('Blunt force, both mechanisms modeled', MM.est(4), len(qm)), ('Strangulation, both mechanisms modeled', MM.est(5), len(qm)),
                  ('Difference between mechanisms', MM.combo(4, 5, -1), len(qm)),
                  ('Strangulation ever, adjusting for total burden', Ms.est(4), len(qs_))):
    row('ST5', [lab] + est_cells(n, t)); out(f'  {lab}', B(t))


print('\n  Reserve x ten or more BIs (binary model) across CNS Vital Signs domains (Supplementary Table 6)')
NAMES = {'composite_memory': 'Composite Memory', 'verbal_memory': 'Verbal Memory', 'visual_memory': 'Visual Memory',
         'working_memory': 'Working Memory', 'cognitive_flexibility': 'Cognitive Flexibility', 'complex_attention': 'Complex Attention',
         'executive_function': 'Executive Function', 'motor_speed': 'Motor Speed', 'processing_speed': 'Processing Speed',
         'psychomotor_speed': 'Psychomotor Speed', 'reaction_time': 'Reaction Time', 'reasoning': 'Reasoning',
         'simple_attention': 'Simple Attention', 'sustained_attention': 'Sustained Attention', 'social_acuity': 'Social Acuity'}
for fam, cols in (('Memory', MEM), ('Non-memory', NON)):
    rr = []
    for col in cols:
        q, M = binary_model(y=col); rr.append((col, len(q), M.est(3)))
    qv = stats.false_discovery_control([x[2][3] for x in rr])
    for (col, n, t), q_ in zip(rr, qv):
        row('ST6', [fam, NAMES[col]] + est_cells(n, t) + [P(q_)]); out(f'  {fam}: {NAMES[col]}', B(t) + f', q = {fp(q_)}, n = {n}')

Yl, rows_ = [], []
for fam, cols in ((1.0, MEM), (0.0, NON)):
    for col in cols:
        q = sample([col, RES, 'ipv_bi_count'] + MEMCOV); v = z(q[col])
        for k_, (_, r_) in enumerate(q.iterrows()):
            Yl.append(v[k_]); rows_.append((r_.ID, fam, r_[RES], float(r_.ipv_bi_count >= 10), r_.education_years, r_.ptsd_pcl5))
R_ = pd.DataFrame(rows_, columns=['ID', 'fam', 'res', 'hi', 'edu', 'ptsd'])
tz_, ez_, pz_ = z(R_.res), z(R_.edu), z(R_.ptsd)
XL = np.column_stack([np.ones(len(R_)), tz_, R_.hi, R_.fam, tz_ * R_.hi, tz_ * R_.fam, R_.hi * R_.fam, tz_ * R_.hi * R_.fam, ez_, pz_])
bL, seL, dfL = ri_lmm(Yl, XL, R_.ID); tL = stats.t.ppf(.975, dfL)
out('  Reserve x injury x domain family (mixed model)',
    B((bL[7], bL[7] - tL * seL[7], bL[7] + tL * seL[7], 2 * stats.t.sf(abs(bL[7] / seL[7]), dfL))) + f', {R_.ID.nunique()} participants')
out('  Reserve-by-injury slope, memory and non-memory domains', f'{bL[4] + bL[7]:.2f} and {bL[4]:.2f}')
qe = sample([Y, 'education_years', 'ipv_bi_count', 'ptsd_pcl5'])
e_ = z(qe.education_years); h_ = (qe.ipv_bi_count >= 10).astype(float).values
out('  Education x ten or more BIs on delayed recall', B(OLS(z(qe[Y]), [e_, h_, e_ * h_, z(qe.ptsd_pcl5)]).est(3)))


qa_ = sample([Y, 'ipv_bi_count'] + MEMCOV); yv = z(qa_[Y]); cv_ = [z(qa_[v]) for v in MEMCOV]; cc = qa_.ipv_bi_count.values.astype(float)


def aic(e):
    M_ = OLS(yv, [e] + cv_); n_ = len(yv); return n_ * math.log(float(M_.resid @ M_.resid) / n_) + 2 * (M_.k + 1)


AIC = {'log(1 + count)': aic(z(np.log1p(np.minimum(cc, 25)))), 'linear count': aic(z(np.minimum(cc, 25))),
       'ordinal (0 / 1–9 / ≥10)': aic(np.where(cc == 0, 0., np.where(cc < 10, 1., 2.))), 'any vs none': aic((cc > 0).astype(float))}
best = min(AIC.values())
print('\n  Model fit by exposure coding, counts capped at 25 (ΔAIC from the best coding; Fig. S1)')
for k in sorted(AIC, key=AIC.get):
    out(f'  {k}', f'{AIC[k] - best:.2f}')

header('SUPPLEMENTARY TABLES 3 TO 6')
EST = ['n', 'β', '95% CI lower', '95% CI upper', 'P']
save_table('supp_table3_sensitivity_reserve_by_injury', ['Specification'] + EST, TABLE_ROWS['ST3'])
save_table('supp_table4_exposure_thresholds', ['Threshold', 'n above', 'n below', 'β', '95% CI lower', '95% CI upper', 'P'], TABLE_ROWS['ST4'])
save_table('supp_table5_injury_mechanism', ['Specification'] + EST, TABLE_ROWS['ST5'])
save_table('supp_table6_cnsvs_domains', ['Family', 'Domain'] + EST + ['q'], TABLE_ROWS['ST6'])
