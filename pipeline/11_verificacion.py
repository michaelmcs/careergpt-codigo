"""Verificaciones de robustez: validación anidada, intervalos y brecha por persona, y validación temporal."""
import os, json, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
src = open('pipeline/03_resultados.py').read()
ns = {'__file__': 'pipeline/03_resultados.py'}
exec(src[:src.index("cv = StratifiedGroupKFold(n_splits=5")], ns)
d, X, y, g, params = ns['d'], ns['X'], ns['y'], ns['g'], ns['params']
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, confusion_matrix
from scipy import stats
from xgboost import XGBClassifier
spw = (len(y) - y.sum()) / y.sum()
GRID = [dict(max_depth=md, learning_rate=lr, n_estimators=ne, min_child_weight=mc, colsample_bytree=cs, subsample=.8, reg_lambda=rl)
        for md, lr, ne, mc, cs, rl in [(2, .05, 400, 20, .6, 5), (3, .03, 500, 20, .6, 5), (3, .05, 300, 50, .5, 10),
                                       (4, .03, 400, 30, .6, 5), (2, .1, 300, 50, .5, 10), (3, .02, 800, 40, .5, 10), (6, .05, 600, 5, .8, 1)]]
mk = lambda gp: XGBClassifier(**gp, scale_pos_weight=spw, enable_categorical=True, tree_method='hist', max_cat_to_onehot=1,
                              random_state=42, n_jobs=-1)
out = {}

# 1. validación cruzada anidada, con los mismos pliegues externos de la tesis
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(y)); fold_auc = []; elegidos = []
for tr, te in cv.split(X, y, g):
    icv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=7)
    best = None
    for gp in GRID:
        a = []
        for itr, ite in icv.split(X.iloc[tr], y[tr], g[tr]):
            m = mk(gp).fit(X.iloc[tr].iloc[itr], y[tr][itr]); a.append(roc_auc_score(y[tr][ite], m.predict_proba(X.iloc[tr].iloc[ite])[:, 1]))
        if best is None or np.mean(a) > best[0]: best = (np.mean(a), gp)
    m = mk(best[1]).fit(X.iloc[tr], y[tr]); p = m.predict_proba(X.iloc[te])[:, 1]
    oof[te] = p; fold_auc.append(roc_auc_score(y[te], p)); elegidos.append(best[1]['max_depth'])
    print('pliegue', len(fold_auc), round(fold_auc[-1], 4), best[1], flush=True)
tt = stats.ttest_1samp(fold_auc, .85, alternative='greater')
out['anidada'] = dict(auc_global=roc_auc_score(y, oof), pliegues=fold_auc, media=float(np.mean(fold_auc)), t=tt.statistic, p=tt.pvalue)
print('ANIDADA', out['anidada'], flush=True)

# 2. bootstrap por persona del AUC con las predicciones de la tesis (no anidadas) y anidadas
oof_t = np.zeros(len(y))
for tr, te in cv.split(X, y, g):
    oof_t[te] = mk(json.load(open('resultados/mejores_params.json'))).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
pers = pd.Series(np.arange(len(y))).groupby(g).apply(np.array).values
rng = np.random.default_rng(42)
def boot(pred, B=1000):
    r = []
    for _ in range(B):
        idx = np.concatenate(pers[rng.integers(0, len(pers), len(pers))]); r.append(roc_auc_score(y[idx], pred[idx]))
    return np.percentile(r, [2.5, 97.5]).tolist()
out['ic_persona_tesis'] = boot(oof_t); out['ic_persona_anidada'] = boot(oof)
out['auc_tesis'] = roc_auc_score(y, oof_t)
print('IC por persona', out['ic_persona_tesis'], out['ic_persona_anidada'], flush=True)

# 3. brecha rural de falsos negativos con bootstrap por persona, umbral F1 de la tesis
from sklearn.metrics import precision_recall_curve
pr_, rc_, uu = precision_recall_curve(y, oof_t); f_ = 2*pr_*rc_/(pr_+rc_+1e-12); u = uu[int(np.nanargmax(f_[:-1]))]; pred = (oof_t >= u).astype(int); area = d.area_colegio.astype(str).values
def tfn(idx):
    r = {}
    for a in ('RURAL', 'URBANA'):
        k = idx[(area[idx] == a) & (y[idx] == 1)]; r[a] = (pred[k] == 0).mean() * 100
    return r['RURAL'] - r['URBANA']
base = tfn(np.arange(len(y))); bb = [tfn(np.concatenate(pers[rng.integers(0, len(pers), len(pers))])) for _ in range(1000)]
out['brecha'] = dict(brecha=base, ic=np.percentile(bb, [2.5, 97.5]).tolist(), p_una_cola=float(np.mean(np.array(bb) <= 0)))
print('BRECHA', out['brecha'], flush=True)

# 4. validación temporal: entrenar 2021-2024 y evaluar 2025
anio = d.anio.astype(int).values; tr = anio <= 2024; te = anio == 2025
mt = mk(json.load(open('resultados/mejores_params.json'))).fit(X[tr], y[tr])
out['temporal_2025'] = roc_auc_score(y[te], mt.predict_proba(X[te])[:, 1])
print('TEMPORAL', out['temporal_2025'], flush=True)
json.dump(out, open('resultados/verificacion.json', 'w'), indent=1, default=float)
