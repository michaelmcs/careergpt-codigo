import os as _os
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import pandas as pd, numpy as np, json, warnings
from pathlib import Path
from scipy import stats
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, precision_score,
                             recall_score, accuracy_score, confusion_matrix, roc_curve,
                             precision_recall_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
S = Path('resultados'); R = {}
RNG = 42

df = pd.read_pickle(S / 'dataset_analitico.pkl')
PER = ['2021-I','2021-II','2022-I','2022-II','2023-I','2023-II','2024-I','2024-II','2025-I','2025-II']
df = df[df.periodo.isin(PER)].copy()
NUM_ALL = [c for c in df.columns if c.endswith('_media')] + ['promedio_secundaria','promedio_5to',
    'tendencia_notas','dispersion_notas','areas_desaprobadas_total','perfil_cuantitativo',
    'perfil_verbal','brecha_cuanti_verbal','tasa_previa','edad_al_examen','anios_desde_egreso',
    'integrantes_hogar','numero_hermanos','numero_postulaciones_anteriores']
for c in NUM_ALL:
    df[c] = pd.to_numeric(df[c], errors='coerce')

R['poblacion'] = dict(postulaciones=len(df), personas=int(df.id_anonimo.nunique()),
    con_historial=int(df.promedio_secundaria.notna().sum()),
    pct_historial=round(df.promedio_secundaria.notna().mean()*100, 1),
    ausentes=int((df.rindio_examen == 0).sum()))
d = df[(df.rindio_examen == 1) & df.promedio_secundaria.notna()].copy()
R['poblacion'].update(analizables=len(d), personas_analizables=int(d.id_anonimo.nunique()),
                      tasa_ingreso=round(d.ingreso.mean()*100, 2),
                      ingresantes=int(d.ingreso.sum()))

# ---------- Tabla: distribución por periodo y proceso
tp = (d.groupby('periodo').agg(postulaciones=('ingreso','size'), ingresantes=('ingreso','sum'))
        .reindex(PER))
tp['tasa'] = (tp.ingresantes/tp.postulaciones*100).round(2)
tp.to_csv(S/'t_periodo.csv')
tpr = d.groupby('tipo_proceso').agg(postulaciones=('ingreso','size'), ingresantes=('ingreso','sum'))
tpr['tasa'] = (tpr.ingresantes/tpr.postulaciones*100).round(2)
tpr.to_csv(S/'t_proceso.csv')

# ---------- P1: variables asociadas al ingreso
filas = []
etiquetas = {'promedio_5to':'Promedio de 5.° de secundaria','promedio_secundaria':'Promedio de secundaria (3.° a 5.°)',
 'matematica_media':'Matemática','comunicacion_media':'Comunicación','ciencia_tecnologia_media':'Ciencia y Tecnología',
 'ciencias_sociales_media':'Ciencias Sociales','ingles_media':'Inglés','tendencia_notas':'Tendencia de notas (5.° menos 3.°)',
 'areas_desaprobadas_total':'Áreas desaprobadas (total)','edad_al_examen':'Edad al examen',
 'anios_desde_egreso':'Años desde el egreso','numero_postulaciones_anteriores':'Postulaciones anteriores',
 'integrantes_hogar':'Integrantes del hogar','tasa_previa':'Tasa de ingreso previa del programa'}
for c, e in etiquetas.items():
    a = d.loc[d.ingreso == 1, c].dropna(); b = d.loc[d.ingreso == 0, c].dropna()
    U, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    r = 1 - 2*U/(len(a)*len(b))  # delta de Cliff con signo invertido
    filas.append(dict(variable=e, ingreso_media=a.mean(), ingreso_de=a.std(), no_media=b.mean(),
                      no_de=b.std(), U=U, p=p, delta=-r))
t_num = pd.DataFrame(filas); t_num.to_csv(S/'t_p1_numericas.csv', index=False)

cats = {'area_colegio':'Área del colegio','tipo_gestion_colegio':'Gestión del colegio','sexo':'Sexo',
        'lengua_materna':'Lengua materna','estudio_cepreuna_previo':'Estudió en CEPREUNA',
        'tipo_preparacion_preuniversitaria':'Preparación preuniversitaria','internet':'Acceso a internet',
        'computadora':'Computadora en el hogar','rango_ingreso_familiar':'Rango de ingreso familiar',
        'nivel_educativo_madre':'Nivel educativo de la madre'}
fc = []
for c, e in cats.items():
    ct = pd.crosstab(d[c].fillna('No registrado'), d.ingreso)
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    v = np.sqrt(chi2/(ct.values.sum()*(min(ct.shape)-1)))
    fc.append(dict(variable=e, chi2=chi2, gl=dof, p=p, V=v))
t_cat = pd.DataFrame(fc).sort_values('V', ascending=False); t_cat.to_csv(S/'t_p1_categoricas.csv', index=False)

# ---------- O1: modelo tabular
NUM = ['edad_al_examen','anios_desde_egreso','integrantes_hogar','numero_hermanos','servicios_basicos',
       'equipamiento_tic','numero_postulaciones_anteriores','promedio_secundaria','promedio_5to',
       'tendencia_notas','dispersion_notas','areas_desaprobadas_total','perfil_cuantitativo',
       'perfil_verbal','brecha_cuanti_verbal','grados_registrados','repitio','se_retiro',
       'grados_requiere_recuperacion','tasa_previa'] + [c for c in d.columns if c.endswith('_media')]
CAT = ['tipo_proceso','area_examen','sexo','estado_civil','lengua_materna','discapacidad_conadis',
       'nivel_educativo_padre','nivel_educativo_madre','ocupacion_padre','ocupacion_madre',
       'rango_ingreso_familiar','tipo_vivienda','trabaja_actualmente','financia_estudios',
       'tipo_gestion_colegio','area_colegio','ugel_colegio','programa_primera_opcion','facultad',
       'area_programa','estudio_cepreuna_previo','tipo_preparacion_preuniversitaria',
       'estudio_universidad_previamente','ie_turno','ie_forma_atencion_modelo_servicio','ie_gestion_dependencia']
NUM = [c for c in NUM if c in d]; CAT = [c for c in CAT if c in d]
X = d[NUM+CAT].copy()
for c in NUM: X[c] = pd.to_numeric(X[c], errors='coerce')
for c in CAT: X[c] = X[c].astype(str).replace('nan', 'No registrado').astype('category')
y = d.ingreso.values; g = d.id_anonimo.values
R['predictores'] = dict(numericos=len(NUM), categoricos=len(CAT))

bp = json.load(open(S/'mejores_params.json'))
params = dict(**bp, scale_pos_weight=(len(y)-y.sum())/y.sum(), enable_categorical=True, max_cat_to_onehot=1,
              tree_method='hist', random_state=RNG, n_jobs=-1)
R['hiperparametros'] = bp
Xlr = X.copy()
for c in CAT: Xlr[c] = Xlr[c].astype(str)
lr = make_pipeline(ColumnTransformer([
        ('n', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), NUM),
        ('c', OneHotEncoder(handle_unknown='ignore', min_frequency=30), CAT)]),
     LogisticRegression(max_iter=2000, class_weight='balanced', C=0.5))

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RNG)
oof_x = np.zeros(len(y)); oof_l = np.zeros(len(y)); pl = []
for k, (tr, te) in enumerate(cv.split(X, y, g), 1):
    mx = XGBClassifier(**params).fit(X.iloc[tr], y[tr]); px = mx.predict_proba(X.iloc[te])[:, 1]
    ml = lr.fit(Xlr.iloc[tr], y[tr]); pL = ml.predict_proba(Xlr.iloc[te])[:, 1]
    oof_x[te] = px; oof_l[te] = pL
    pl.append(dict(pliegue=k, n=len(te), auc_xgb=roc_auc_score(y[te], px), aucpr_xgb=average_precision_score(y[te], px),
                   auc_lr=roc_auc_score(y[te], pL), aucpr_lr=average_precision_score(y[te], pL)))
t_folds = pd.DataFrame(pl); t_folds.to_csv(S/'t_pliegues.csv', index=False)

def umbral_f1(p):
    pr, rc, u = precision_recall_curve(y, p); f = 2*pr*rc/(pr+rc+1e-12); return u[int(np.nanargmax(f[:-1]))]
def met(p, nombre):
    u = umbral_f1(p); pr = (p >= u).astype(int)
    return dict(modelo=nombre, auc=roc_auc_score(y, p), aucpr=average_precision_score(y, p),
                exactitud=accuracy_score(y, pr), precision=precision_score(y, pr),
                sensibilidad=recall_score(y, pr), f1=f1_score(y, pr), umbral=u)
trivial = dict(modelo='Clasificador trivial (siempre no ingresa)', auc=0.5, aucpr=y.mean(),
               exactitud=1-y.mean(), precision=0.0, sensibilidad=0.0, f1=0.0, umbral=np.nan)
t_modelos = pd.DataFrame([trivial, met(oof_l, 'Regresión logística'), met(oof_x, 'XGBoost')])
t_modelos.to_csv(S/'t_modelos.csv', index=False)

# bootstrap del AUC y prueba de H1
rng = np.random.default_rng(RNG); bs = []
for _ in range(1000):
    i = rng.integers(0, len(y), len(y))
    if y[i].sum() > 0: bs.append(roc_auc_score(y[i], oof_x[i]))
ic = np.percentile(bs, [2.5, 97.5])
tt = stats.ttest_1samp(t_folds.auc_xgb, 0.85, alternative='greater')
# comparación XGBoost frente a regresión logística por pliegue
tw = stats.ttest_rel(t_folds.auc_xgb, t_folds.auc_lr)
R['h1'] = dict(auc=roc_auc_score(y, oof_x), ic_inf=ic[0], ic_sup=ic[1], t=tt.statistic, p=tt.pvalue,
               auc_media=t_folds.auc_xgb.mean(), auc_de=t_folds.auc_xgb.std(),
               aucpr=average_precision_score(y, oof_x),
               t_vs_lr=tw.statistic, p_vs_lr=tw.pvalue,
               mejora_vs_lr=(t_folds.auc_xgb.mean()-t_folds.auc_lr.mean()))
u = umbral_f1(oof_x); pred = (oof_x >= u).astype(int)
tn, fp, fn, tp_ = confusion_matrix(y, pred).ravel()
R['confusion'] = dict(VN=int(tn), FP=int(fp), FN=int(fn), VP=int(tp_), umbral=float(u))

final = XGBClassifier(**params).fit(X, y)
bst = final.get_booster(); gain = bst.get_score(importance_type='gain')
imp = pd.DataFrame({'variable': list(gain), 'ganancia': list(gain.values())})
imp['ganancia_rel'] = imp.ganancia/imp.ganancia.sum()*100
imp = imp.sort_values('ganancia_rel', ascending=False); imp.to_csv(S/'t_importancia.csv', index=False)

# equidad
d['p'] = oof_x; d['pred'] = pred; eq = []
for gr, s in d.groupby('area_colegio'):
    a, b, c_, e_ = confusion_matrix(s.ingreso, s.pred, labels=[0, 1]).ravel()
    eq.append(dict(grupo=gr, n=len(s), tasa_real=s.ingreso.mean()*100, auc=roc_auc_score(s.ingreso, s.p),
                   tfn=c_/(c_+e_)*100, tfp=b/(b+a)*100, fn=c_, vp=e_))
t_eq = pd.DataFrame(eq); t_eq.to_csv(S/'t_equidad.csv', index=False)
ctab = np.array([[r.fn, r.vp] for r in t_eq.itertuples()])
chi, pe, _, _ = stats.chi2_contingency(ctab)
R['equidad'] = dict(chi2=chi, p=pe, brecha_tfn=abs(t_eq.tfn.iloc[0]-t_eq.tfn.iloc[1]))

# ---------- figuras
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 10})
fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
for p_, n_, ls in [(oof_x, 'XGBoost', '-'), (oof_l, 'Regresión logística', '--')]:
    f_, t_, _ = roc_curve(y, p_); ax[0].plot(f_, t_, ls, color=('#2E75B6' if n_=='XGBoost' else '#C55A11'), lw=1.8, label=f'{n_} (AUC = {roc_auc_score(y, p_):.3f})')
    pr, rc, _ = precision_recall_curve(y, p_); ax[1].plot(rc, pr, ls, color=('#2E75B6' if n_=='XGBoost' else '#C55A11'), lw=1.8, label=f'{n_} (AP = {average_precision_score(y, p_):.3f})')
ax[0].plot([0, 1], [0, 1], ':', color='#7F7F7F', lw=1); [a.grid(alpha=.25) for a in ax]; ax[1].axhline(y.mean(), ls=':', color='gray', lw=1, label=f'Tasa base ({y.mean():.3f})')
ax[0].set(xlabel='Tasa de falsos positivos', ylabel='Tasa de verdaderos positivos', title='a) Curva ROC')
ax[1].set(xlabel='Sensibilidad', ylabel='Precisión', title='b) Curva de precisión y sensibilidad')
for a in ax: a.legend(fontsize=8, frameon=False)
plt.tight_layout(); plt.savefig(S/'fig_roc_pr.png', dpi=200); plt.close()

nombres = {'promedio_5to': 'Promedio de 5.°', 'equipamiento_tic': 'Equipamiento TIC', 'promedio_secundaria': 'Promedio de secundaria',
           'tasa_previa': 'Tasa previa del programa', 'tipo_proceso': 'Tipo de proceso',
           'tipo_preparacion_preuniversitaria': 'Preparación preuniversitaria', 'programa_primera_opcion': 'Programa postulado',
           'ugel_colegio': 'UGEL del colegio', 'facultad': 'Facultad', 'estudio_cepreuna_previo': 'Estudió en CEPREUNA',
           'numero_postulaciones_anteriores': 'Postulaciones anteriores', 'rango_ingreso_familiar': 'Ingreso familiar',
           'perfil_cuantitativo': 'Perfil cuantitativo', 'ie_gestion_dependencia': 'Dependencia de la IE', 'perfil_verbal': 'Perfil verbal',
           'area_programa': 'Área del programa', 'area_examen': 'Área del examen', 'matematica_media': 'Matemática',
           'computadora': 'Computadora', 'internet': 'Internet', 'area_colegio': 'Área del colegio'}
top = imp.head(12).iloc[::-1]
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.barh([nombres.get(v, v) for v in top.variable], top.ganancia_rel, color='0.45', edgecolor='black', lw=.5)
ax.set_xlabel('Ganancia relativa (%)'); plt.tight_layout(); plt.savefig(S/'fig_importancia.png', dpi=200); plt.close()
imp['etiqueta'] = imp.variable.map(lambda v: nombres.get(v, v)); imp.to_csv(S/'t_importancia.csv', index=False)

# ---------- calibración de probabilidades
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
iso = IsotonicRegression(out_of_bounds='clip').fit(oof_x, y)
R['calibracion'] = dict(brier_sin=brier_score_loss(y, oof_x), brier_con=brier_score_loss(y, iso.predict(oof_x)),
                        media_sin=float(oof_x.mean()), media_con=float(iso.predict(oof_x).mean()), tasa=float(y.mean()))
# ---------- O2: corpus de instrucción y respuesta
def nivel(x):
    if pd.isna(x): return 'sin registro'
    return 'alto desempeño' if x >= 16 else 'desempeño satisfactorio' if x >= 13 else 'desempeño en proceso' if x >= 11 else 'desempeño en inicio'
progs = d.groupby('area_programa').programa_primera_opcion.agg(lambda s: sorted(s.value_counts().index[:10]))
ult = d.sort_values(['anio', 'semestre']).groupby('id_anonimo').tail(1).copy()
Xu = X.loc[ult.index]
pares, ranks = [], []
for area, plist in progs.items():
    idx = ult.index[ult.area_programa == area]
    if len(idx) == 0: continue
    base = Xu.loc[idx]; sc = []
    for pg in plist:
        tmp = base.copy(); tmp['programa_primera_opcion'] = pd.Categorical([pg]*len(tmp), categories=X.programa_primera_opcion.cat.categories)
        if 'facultad' in tmp:
            fac = d.loc[d.programa_primera_opcion == pg, 'facultad'].mode()
            if len(fac): tmp['facultad'] = pd.Categorical([str(fac.iloc[0])]*len(tmp), categories=X.facultad.cat.categories)
        sc.append(iso.predict(final.predict_proba(tmp)[:, 1]))
    sc = np.vstack(sc).T
    for j, ix in enumerate(idx):
        o = np.argsort(-sc[j])[:3]; ranks.append((ix, [(plist[k], float(sc[j][k])) for k in o]))
AR = {'matematica_media': 'Matemática', 'comunicacion_media': 'Comunicación', 'ciencia_tecnologia_media': 'Ciencia y Tecnología',
      'ciencias_sociales_media': 'Ciencias Sociales', 'ingles_media': 'Inglés'}
for ix, top3 in ranks:
    r = d.loc[ix]; notas = {AR[k]: r[k] for k in AR if pd.notna(r[k])}
    fuerte = max(notas, key=notas.get) if notas else 'sin registro'; debil = min(notas, key=notas.get) if notas else 'sin registro'
    instr = (f"Perfil del postulante: procede de un colegio {'público' if str(r.tipo_gestion_colegio).upper().startswith('PÚB') else 'privado'} de área {str(r.area_colegio).lower()} "
             f"({r.ugel_colegio}); promedio de secundaria con {nivel(r.promedio_secundaria)}; mayor fortaleza en {fuerte} y menor en {debil}; "
             f"postulaciones anteriores: {int(r.numero_postulaciones_anteriores) if pd.notna(r.numero_postulaciones_anteriores) else 0}; "
             f"preparación preuniversitaria: {str(r.tipo_preparacion_preuniversitaria).lower()}; área de interés: {str(r.area_programa).lower()}. "
             f"Probabilidades estimadas de ingreso: " + '; '.join(f'{p} {s*100:.1f} %' for p, s in top3) +
             ". Genera una ruta educativa personalizada.")
    resp = (f"Ruta educativa sugerida. Primera opción recomendada: {top3[0][0]}, con la mayor probabilidad estimada de ingreso "
            f"({top3[0][1]*100:.1f} %). Alternativas: {top3[1][0]} ({top3[1][1]*100:.1f} %) y {top3[2][0]} ({top3[2][1]*100:.1f} %). "
            f"Tu fortaleza en {fuerte} es coherente con estas opciones. Para mejorar tus posibilidades, refuerza {debil} antes del examen"
            f"{' y considera una preparación preuniversitaria estructurada' if str(r.tipo_preparacion_preuniversitaria).upper().startswith('NING') else ''}. "
            f"Estas probabilidades son estimaciones orientativas basadas en postulantes con perfiles similares, no una garantía de resultado.")
    pares.append(dict(id=r.id_anonimo, area=str(r.area_programa), area_colegio=str(r.area_colegio), sexo=str(r.sexo),
                      instruction=instr, output=resp, top1=top3[0][0]))
P = pd.DataFrame(pares)
gss = GroupShuffleSplit(n_splits=1, train_size=.70, random_state=RNG); tr, rest = next(gss.split(P, groups=P.id))
gss2 = GroupShuffleSplit(n_splits=1, train_size=.50, random_state=RNG); va, te = next(gss2.split(P.iloc[rest], groups=P.iloc[rest].id))
P['particion'] = 'entrenamiento'; P.loc[P.index[rest[va]], 'particion'] = 'validacion'; P.loc[P.index[rest[te]], 'particion'] = 'prueba'
with open(S/'CareER_Dataset.jsonl', 'w', encoding='utf8') as f:
    for r in P.itertuples():
        f.write(json.dumps({'instruction': r.instruction, 'input': '', 'output': r.output, 'split': r.particion}, ensure_ascii=False)+'\n')
P['long_in'] = P.instruction.str.split().str.len(); P['long_out'] = P.output.str.split().str.len()
t_part = P.groupby('particion').agg(pares=('id', 'size'), palabras_instr=('long_in', 'mean'), palabras_resp=('long_out', 'mean')).reindex(['entrenamiento', 'validacion', 'prueba'])
t_part.to_csv(S/'t_corpus.csv')
# representatividad frente a la población analizable (personas)
pop = ult
rep = []
for c, n in [('area_colegio', 'Área del colegio'), ('sexo', 'Sexo'), ('area_programa', 'Área del programa')]:
    obs = P[c if c in P else 'area'].value_counts() if c != 'area_programa' else P.area.value_counts()
    ref = pop[c].astype(str).value_counts(normalize=True)
    obs = obs.reindex(ref.index).fillna(0); exp = ref*obs.sum()
    ch, pp = stats.chisquare(obs, exp)
    # balance entre particiones
    ct = pd.crosstab(P[c] if c in P else P.area, P.particion) if c != 'area_programa' else pd.crosstab(P.area, P.particion)
    ch2, pp2, _, _ = stats.chi2_contingency(ct)
    rep.append(dict(variable=n, chi2_pob=ch, p_pob=pp, chi2_part=ch2, p_part=pp2))
t_rep = pd.DataFrame(rep); t_rep.to_csv(S/'t_representatividad.csv', index=False)
coh = (P.apply(lambda r: r.top1 in r.output and r.top1 in r.instruction, axis=1)).mean()*100
R['corpus'] = dict(pares=len(P), coherencia_top1=coh, vocab=int(len(set(' '.join(P.output).split()))),
                   top1_distintos=int(P.top1.nunique()))
R['ejemplo_par'] = pares[0]
(S/'resultados.json').write_text(json.dumps(R, indent=2, ensure_ascii=False, default=float))
print(json.dumps(R, indent=1, ensure_ascii=False, default=float)[:3500])
print(t_modelos.round(4).to_string()); print(t_folds.round(4).to_string()); print(t_eq.round(3).to_string())
print(t_num.round(4).to_string()); print(t_cat.round(4).to_string()); print(t_part); print(t_rep.round(4))
print(tp); print(tpr); print(imp.head(12)[['etiqueta', 'ganancia_rel']].round(2).to_string())
