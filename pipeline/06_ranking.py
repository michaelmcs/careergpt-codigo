import os as _os
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import numpy as np, pandas as pd, json, warnings
from scipy import stats
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')
src = open('pipeline/03_resultados.py').read(); ns = {'__file__': __file__}
exec(src[:src.index("bp = json.load")], ns)
X, y, g, d, S, CAT = ns['X'], ns['y'], ns['g'], ns['d'], ns['S'], ns['CAT']
bp = json.load(open(S / 'mejores_params.json'))
params = dict(**bp, scale_pos_weight=(len(y) - y.sum()) / y.sum(), enable_categorical=True, max_cat_to_onehot=1,
              tree_method='hist', random_state=42, n_jobs=-1)
cats = X.programa_primera_opcion.cat.categories
fac_de = d.dropna(subset=['facultad']).groupby('programa_primera_opcion').facultad.agg(lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0]).to_dict()
cand_area = d.groupby('area_programa').programa_primera_opcion.agg(lambda s: list(s.value_counts().index[:10])).to_dict()
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
filas = []
for k, (tr, te) in enumerate(cv.split(X, y, g), 1):
    m = XGBClassifier(**params).fit(X.iloc[tr], y[tr])
    # popularidad: tasa de ingreso del programa en el pliegue de entrenamiento
    dtr = d.iloc[tr]; pop = dtr.groupby('programa_primera_opcion').ingreso.mean().to_dict()
    ing = np.where(y[te] == 1)[0]
    for j in ing:
        idx = te[j]; fila = d.iloc[idx]; real = str(fila.programa_primera_opcion)
        cands = list(dict.fromkeys(cand_area.get(fila.area_programa, []) + [real]))
        base = X.iloc[[idx] * len(cands)].copy()
        base['programa_primera_opcion'] = pd.Categorical(cands, categories=cats)
        if 'facultad' in base:
            base['facultad'] = pd.Categorical([str(fac_de.get(c, fila.facultad)) for c in cands], categories=X.facultad.cat.categories)
        s_x = m.predict_proba(base)[:, 1]
        s_p = np.array([pop.get(c, 0) for c in cands])
        r_x = 1 + (s_x > s_x[cands.index(real)]).sum()
        r_p = 1 + (s_p > s_p[cands.index(real)]).sum()
        filas.append(dict(pliegue=k, area_colegio=str(fila.area_colegio), m=len(cands), rank_xgb=r_x, rank_pop=r_p))
    print('pliegue', k, 'ingresantes evaluados', len(ing), flush=True)
R = pd.DataFrame(filas)
def met(r, m):
    return dict(MRR=np.mean(1 / r), NDCG5=np.mean(np.where(r <= 5, 1 / np.log2(r + 1), 0)),
                NDCG10=np.mean(np.where(r <= 10, 1 / np.log2(r + 1), 0)), Hit1=np.mean(r == 1), Hit3=np.mean(r <= 3))
# aleatorio: valores esperados exactos
def aleatorio(mm):
    out = {'MRR': [], 'NDCG5': [], 'NDCG10': [], 'Hit1': [], 'Hit3': []}
    for m_ in mm:
        rs = np.arange(1, m_ + 1)
        out['MRR'].append(np.mean(1 / rs)); out['NDCG5'].append(np.mean(np.where(rs <= 5, 1 / np.log2(rs + 1), 0)))
        out['NDCG10'].append(np.mean(np.where(rs <= 10, 1 / np.log2(rs + 1), 0)))
        out['Hit1'].append(1 / m_); out['Hit3'].append(min(3, m_) / m_)
    return {k: float(np.mean(v)) for k, v in out.items()}
T = pd.DataFrame([dict(metodo='Aleatorio', **aleatorio(R.m)), dict(metodo='Popularidad del programa', **met(R.rank_pop, R.m)),
                  dict(metodo='XGBoost', **met(R.rank_xgb, R.m))])
w = stats.wilcoxon(1 / R.rank_xgb, 1 / R.rank_pop)
mejora = (T.MRR.iloc[2] - T.MRR.iloc[1]) / T.MRR.iloc[1] * 100
eqr = R.groupby('area_colegio').apply(lambda s: pd.Series(met(s.rank_xgb, s.m)), include_groups=False)
T.to_csv(S / 't_ranking.csv', index=False); eqr.to_csv(S / 't_ranking_equidad.csv')
res = dict(n=len(R), candidatos_medio=float(R.m.mean()), wilcoxon_W=float(w.statistic), wilcoxon_p=float(w.pvalue), mejora_mrr=mejora,
           r_efecto=float(abs(stats.norm.ppf(w.pvalue / 2)) / np.sqrt(len(R))))
json.dump(res, open(S / 'ranking.json', 'w'), indent=2)
print(T.round(4).to_string()); print(res); print(eqr.round(4))
