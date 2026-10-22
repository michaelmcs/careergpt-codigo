"""
CareER-GPT | Etapa 1: preprocesamiento y construcción del dataset analítico
Une la base de Admisión con los registros escolares del SIAGIE y deja una
tabla con una fila por postulación, usando únicamente información anterior
al examen de admisión.
"""
import os as _os
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import pandas as pd
import numpy as np
import json
from pathlib import Path

ENTRADA = Path('datos')
SALIDA = Path('resultados')
SALIDA.mkdir(parents=True, exist_ok=True)

AREAS = ['matematica', 'comunicacion', 'ciencia_tecnologia', 'ciencias_sociales',
         'dpcc', 'ingles', 'arte_cultura', 'educacion_fisica',
         'educacion_religiosa', 'educacion_trabajo']

# Variables que ocurren durante o después del examen: nunca son predictoras
FUGA = ['puntaje_total', 'puntajes_componentes_json', 'respuestas_por_curso_json',
        'carrera_ingreso', 'orden_merito_carrera', 'orden_merito_general']

print('=' * 70)
print('ETAPA 1: PREPROCESAMIENTO')
print('=' * 70)

# ---------------------------------------------------------------- carga
adm = pd.read_excel(ENTRADA / 'ADMISION_V3.xlsx')
sia = pd.read_excel(ENTRADA / 'SIAGIE_V3.xlsx')
print(f'\nAdmisión : {adm.shape[0]:,} filas x {adm.shape[1]} columnas')
print(f'SIAGIE   : {sia.shape[0]:,} filas x {sia.shape[1]} columnas')

# ------------------------------------------- SIAGIE: de formato largo a ancho
# Si un estudiante repitió, hay más de un registro por grado.
# Se conserva el último año cursado de cada grado, que es el resultado final.
sia = sia.sort_values(['id_anonimo', 'grado', 'anio_lectivo'])
rep = sia.duplicated(['id_anonimo', 'grado'], keep=False).sum()
print(f'\nRegistros escolares en grados repetidos: {rep:,}')
sia_u = sia.drop_duplicates(['id_anonimo', 'grado'], keep='last').copy()

sia_u['g'] = sia_u.grado.str[0]  # 3, 4, 5

cols_grado = [f'{a}_norm_20' for a in AREAS] + [
    'promedio_general_anual_norm_20', 'numero_areas_aprobadas',
    'numero_areas_desaprobadas', 'situacion_final_grado', 'anio_lectivo']

ancho = sia_u.pivot(index='id_anonimo', columns='g', values=cols_grado)
ancho.columns = [f'{c}_g{g}' for c, g in ancho.columns]
ancho = ancho.reset_index()

# Atributos del estudiante y del colegio: se toman del último grado cursado
perfil = (sia_u.sort_values(['id_anonimo', 'g'])
          .groupby('id_anonimo')
          .last()[['codigo_modular', 'nombre_institucion_educativa', 'ugel',
                   'tipo_gestion', 'gestion_dependencia', 'area_geografica',
                   'ubigeo_distrito_ie', 'turno', 'forma_atencion_modelo_servicio',
                   'lengua_materna', 'escala_calificacion']]
          .add_prefix('ie_').reset_index())

trayect = sia_u.groupby('id_anonimo').agg(
    grados_registrados=('g', 'count'),
    anio_escolar_min=('anio_lectivo', 'min'),
    anio_escolar_max=('anio_lectivo', 'max')).reset_index()

# Repitencia y retiro a partir del registro completo, no del deduplicado
eventos = sia.groupby('id_anonimo').agg(
    repitio=('situacion_final_grado', lambda x: int((x == 'REPITE').any())),
    se_retiro=('situacion_final_grado', lambda x: int((x == 'RETIRADO').any())),
    grados_requiere_recuperacion=('situacion_final_grado',
                                  lambda x: int((x == 'REQUIERE RECUPERACIÓN').sum()))
).reset_index()

escolar = (ancho.merge(perfil, on='id_anonimo')
                .merge(trayect, on='id_anonimo')
                .merge(eventos, on='id_anonimo'))
print(f'Estudiantes con historial escolar: {len(escolar):,}')

# ------------------------------------------------------------------ cruce
df = adm.merge(escolar, on='id_anonimo', how='left', indicator=True)
cruzan = (df._merge == 'both').sum()
print(f'\nPostulaciones con historial escolar : {cruzan:,} '
      f'({cruzan / len(df) * 100:.1f} %)')
print(f'Postulaciones sin historial escolar : {(df._merge == "left_only").sum():,}')
df = df.drop(columns='_merge')

# ------------------------------------------------- variables construidas
# Rendimiento escolar consolidado
for a in AREAS:
    cols = [f'{a}_norm_20_g{g}' for g in '345' if f'{a}_norm_20_g{g}' in df]
    df[f'{a}_media'] = df[cols].mean(axis=1)

prom_cols = [f'promedio_general_anual_norm_20_g{g}' for g in '345']
df['promedio_secundaria'] = df[prom_cols].mean(axis=1)
df['promedio_5to'] = df['promedio_general_anual_norm_20_g5']
df['tendencia_notas'] = (df['promedio_general_anual_norm_20_g5']
                         - df['promedio_general_anual_norm_20_g3'])
df['dispersion_notas'] = df[prom_cols].std(axis=1)
df['areas_desaprobadas_total'] = df[[f'numero_areas_desaprobadas_g{g}'
                                     for g in '345']].sum(axis=1, min_count=1)

# Perfil cuantitativo frente a verbal
df['perfil_cuantitativo'] = df[['matematica_media', 'ciencia_tecnologia_media']].mean(axis=1)
df['perfil_verbal'] = df[['comunicacion_media', 'ciencias_sociales_media']].mean(axis=1)
df['brecha_cuanti_verbal'] = df['perfil_cuantitativo'] - df['perfil_verbal']

# Trayectoria temporal
df['anios_desde_egreso'] = df['anio'] - df['anio_egreso_secundaria']

# Contexto socioeconómico agregado
for c in ['agua', 'desague', 'electricidad', 'internet', 'computadora']:
    df[f'_s_{c}'] = (df[c].astype(str).str.upper().isin(['SÍ', 'SI', 'TRUE', '1'])).astype(int)
df['servicios_basicos'] = df[[f'_s_{c}' for c in
                              ['agua', 'desague', 'electricidad']]].sum(axis=1)
df['equipamiento_tic'] = df[['_s_internet', '_s_computadora']].sum(axis=1)
df = df.drop(columns=[c for c in df.columns if c.startswith('_s_')])

# Competencia del programa: se calcula solo con procesos anteriores
df = df.sort_values(['anio', 'semestre'])
df['clave_concurso'] = (df.periodo.astype(str) + '|' + df.tipo_proceso.astype(str)
                        + '|' + df.programa_primera_opcion.astype(str))
hist = (df.groupby(['tipo_proceso', 'programa_primera_opcion', 'periodo'])
          .agg(n_post=('id_anonimo', 'size'),
               n_ing=('condicion_final', lambda x: (x == 'INGRESÓ').sum()))
          .reset_index().sort_values('periodo'))
hist['tasa_previa'] = (hist.groupby(['tipo_proceso', 'programa_primera_opcion'])
                       .apply(lambda g: (g.n_ing.shift().cumsum()
                                         / g.n_post.shift().cumsum()),
                              include_groups=False).reset_index(level=[0, 1], drop=True))
df = df.merge(hist[['tipo_proceso', 'programa_primera_opcion', 'periodo', 'tasa_previa']],
              on=['tipo_proceso', 'programa_primera_opcion', 'periodo'], how='left')

# ------------------------------------------------------- variable objetivo
df['ingreso'] = (df.condicion_final == 'INGRESÓ').astype(int)
df['rindio_examen'] = (df.condicion_final != 'AUSENTE').astype(int)

print('\nDistribución del resultado:')
for k, v in df.condicion_final.value_counts().items():
    print(f'   {k:<14} {v:>7,}  ({v / len(df) * 100:5.2f} %)')

# --------------------------------------------------------------- guardado
df.to_pickle(SALIDA / 'dataset_analitico.pkl')

meta = {
    'filas': int(len(df)),
    'personas': int(df.id_anonimo.nunique()),
    'tasa_emparejamiento': round(cruzan / len(df) * 100, 2),
    'tasa_ingreso': round(df.ingreso.mean() * 100, 2),
    'variables_excluidas_por_fuga': FUGA,
    'periodos': sorted(df.periodo.unique().tolist()),
    'procesos': sorted(df.tipo_proceso.unique().tolist()),
}
(SALIDA / 'metadatos.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False))

print(f'\nDataset analítico: {df.shape[0]:,} filas x {df.shape[1]} columnas')
print(f'Guardado en {SALIDA / "dataset_analitico.pkl"}')
