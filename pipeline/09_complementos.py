"""
CareER-GPT | Complementos de resultados
Caracterización de las postulaciones analizadas, tasa de ingreso según el promedio de quinto de secundaria y
curva de calibración del modelo XGBoost antes y después de la calibración isotónica.
"""
import os as _os
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import json, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

src = open('pipeline/03_resultados.py', encoding='utf8').read(); ns = {'__file__': __file__}
exec(src[:src.index("bp = json.load")], ns)
X, y, g, d, S = ns['X'], ns['y'], ns['g'], ns['d'], ns['S']
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 10})

# ---------------------------------------------------------------- caracterización de las postulaciones
VARS = [('sexo', 'Sexo'), ('area_colegio', 'Área del colegio'), ('tipo_gestion_colegio', 'Gestión del colegio'),
        ('lengua_materna', 'Lengua materna'), ('tipo_preparacion_preuniversitaria', 'Preparación preuniversitaria'),
        ('area_programa', 'Área del programa postulado')]
filas = []
for c, et in VARS:
    s = d[c].fillna('No registrado').astype(str).str.strip().replace({'': 'No registrado', 'nan': 'No registrado'})
    t = pd.DataFrame({'cat': s, 'ingreso': d.ingreso}).groupby('cat').agg(n=('ingreso', 'size'), ing=('ingreso', 'sum'))
    t = t.sort_values('n', ascending=False)
    for cat, r in t.iterrows():
        filas.append(dict(variable=et, categoria=cat, n=int(r.n), pct=r.n / len(d) * 100, tasa=r.ing / r.n * 100))
pd.DataFrame(filas).to_csv(S / 't_caracterizacion.csv', index=False)

# ---------------------------------------------------------------- tasa de ingreso por tramo del promedio de 5.°
# los extremos con pocos casos se agrupan para que la tasa sea estable
cortes = [0, 12, 13, 14, 15, 16, 21]
etq = ['Menos de 12', '12', '13', '14', '15', '16 o más']
tr = pd.cut(d.promedio_5to, cortes, right=False, labels=etq)
tt = (pd.DataFrame({'tramo': tr, 'ingreso': d.ingreso}).dropna()
        .groupby('tramo', observed=False).agg(n=('ingreso', 'size'), tasa=('ingreso', 'mean')))
tt['tasa'] *= 100
tt.to_csv(S / 't_tasa_promedio.csv')
fig, ax = plt.subplots(figsize=(7, 3.6))
b = ax.bar(tt.index.astype(str), tt.tasa, color='#2E75B6', edgecolor='white')
for rect, (_, r) in zip(b, tt.iterrows()):
    ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.4, f'{r.tasa:.1f}\nn = {int(r.n):,}'.replace(',', ' '),
            ha='center', va='bottom', fontsize=7.5)
ax.axhline(d.ingreso.mean() * 100, ls=':', color='#7F7F7F', lw=1)
ax.text(-0.4, d.ingreso.mean() * 100 + 0.8, f'Tasa global ({d.ingreso.mean() * 100:.2f} %)', ha='left',
        fontsize=8, color='#595959')
ax.set_xlabel('Promedio de 5.° de secundaria (escala vigesimal)'); ax.set_ylabel('Tasa de ingreso (%)')
ax.set_ylim(0, tt.tasa.max() * 1.3); ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.savefig(S / 'fig_tasa_promedio.png', dpi=220); plt.close()

# ---------------------------------------------------------------- calibración del modelo XGBoost
bp = json.load(open(S / 'mejores_params.json'))
params = dict(**bp, scale_pos_weight=(len(y) - y.sum()) / y.sum(), enable_categorical=True, max_cat_to_onehot=1,
              tree_method='hist', random_state=42, n_jobs=-1)
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(y))
for tr_, te_ in cv.split(X, y, g):
    oof[te_] = XGBClassifier(**params).fit(X.iloc[tr_], y[tr_]).predict_proba(X.iloc[te_])[:, 1]
R = json.load(open(S / 'resultados.json'))
assert abs(roc_auc_score(y, oof) - R['h1']['auc']) < 1e-9, 'las predicciones no coinciden con 03_resultados.py'
cal = IsotonicRegression(out_of_bounds='clip').fit(oof, y).predict(oof)
fig, ax = plt.subplots(figsize=(5.6, 4.4))
for p_, et, col, mk in [(oof, f'Sin calibrar (Brier = {brier_score_loss(y, oof):.3f})', '#C55A11', 'o'),
                        (cal, f'Calibración isotónica (Brier = {brier_score_loss(y, cal):.3f})', '#2E75B6', 's')]:
    fr, mp = calibration_curve(y, p_, n_bins=10, strategy='quantile')
    ax.plot(mp, fr, marker=mk, ms=4, lw=1.6, color=col, label=et)
ax.plot([0, 1], [0, 1], ':', color='#7F7F7F', lw=1, label='Calibración perfecta')
ax.set_xlabel('Probabilidad estimada'); ax.set_ylabel('Proporción observada de ingreso')
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=.25); ax.legend(frameon=False, fontsize=8, loc='upper left')
plt.tight_layout(); plt.savefig(S / 'fig_calibracion.png', dpi=220); plt.close()
fr_s, mp_s = calibration_curve(y, oof, n_bins=10, strategy='quantile')
json.dump(dict(auc_verificado=float(roc_auc_score(y, oof)), brier_sin=float(brier_score_loss(y, oof)),
               brier_con=float(brier_score_loss(y, cal)), decil_sup_estimada=float(mp_s[-1]),
               decil_sup_observada=float(fr_s[-1])), open(S / 'complementos.json', 'w'), indent=2)
print(pd.DataFrame(filas).round(2).to_string()); print(tt.round(2))
