"""
CareER-GPT | Diccionario de datos para los anexos
Describe cada variable que usa el modelo tabular, su fuente, su tipo y sus valores en el conjunto analítico.
"""
import os as _os
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

src = open('pipeline/03_resultados.py', encoding='utf8').read(); ns = {'__file__': __file__}
exec(src[:src.index("bp = json.load")], ns)
d, NUM, CAT, S = ns['d'], ns['NUM'], ns['CAT'], ns['S']

AREAS = {'matematica': 'Matemática', 'comunicacion': 'Comunicación', 'ciencia_tecnologia': 'Ciencia y Tecnología',
         'ciencias_sociales': 'Ciencias Sociales', 'dpcc': 'Desarrollo Personal, Ciudadanía y Cívica', 'ingles': 'Inglés',
         'arte_cultura': 'Arte y Cultura', 'educacion_fisica': 'Educación Física',
         'educacion_religiosa': 'Educación Religiosa', 'educacion_trabajo': 'Educación para el Trabajo'}
DESC = {
    'edad_al_examen': ('Edad del postulante a la fecha del examen, en años', 'Admisión'),
    'anios_desde_egreso': ('Años transcurridos desde el egreso de la educación secundaria', 'Derivada'),
    'integrantes_hogar': ('Número de integrantes del hogar', 'Admisión'),
    'numero_hermanos': ('Número de hermanos', 'Admisión'),
    'servicios_basicos': ('Índice de acceso a agua, desagüe y electricidad', 'Derivada'),
    'equipamiento_tic': ('Índice de acceso a internet y computadora en el hogar', 'Derivada'),
    'numero_postulaciones_anteriores': ('Número de postulaciones previas a la universidad', 'Admisión'),
    'promedio_secundaria': ('Promedio general anual de tercero a quinto de secundaria', 'Derivada del SIAGIE'),
    'promedio_5to': ('Promedio general anual de quinto de secundaria', 'SIAGIE'),
    'tendencia_notas': ('Promedio de quinto grado menos promedio de tercer grado', 'Derivada del SIAGIE'),
    'dispersion_notas': ('Desviación estándar de los promedios de tercero a quinto grado', 'Derivada del SIAGIE'),
    'areas_desaprobadas_total': ('Total de áreas desaprobadas de tercero a quinto grado', 'Derivada del SIAGIE'),
    'perfil_cuantitativo': ('Media de Matemática y Ciencia y Tecnología', 'Derivada del SIAGIE'),
    'perfil_verbal': ('Media de Comunicación y Ciencias Sociales', 'Derivada del SIAGIE'),
    'brecha_cuanti_verbal': ('Perfil cuantitativo menos perfil verbal', 'Derivada del SIAGIE'),
    'grados_registrados': ('Número de grados de tercero a quinto con registro', 'Derivada del SIAGIE'),
    'repitio': ('Indicador de repitencia de algún grado, 1 si repitió', 'Derivada del SIAGIE'),
    'se_retiro': ('Indicador de retiro en algún grado, 1 si se retiró', 'Derivada del SIAGIE'),
    'grados_requiere_recuperacion': ('Número de grados con situación final de recuperación', 'Derivada del SIAGIE'),
    'tasa_previa': ('Tasa de ingreso del programa en procesos anteriores del mismo tipo', 'Derivada de Admisión'),
    'tipo_proceso': ('Modalidad del proceso de admisión', 'Admisión'),
    'area_examen': ('Área académica del examen', 'Admisión'),
    'sexo': ('Sexo del postulante', 'Admisión'),
    'estado_civil': ('Estado civil del postulante', 'Admisión'),
    'lengua_materna': ('Lengua materna declarada', 'Admisión'),
    'discapacidad_conadis': ('Registro de discapacidad', 'Admisión'),
    'nivel_educativo_padre': ('Nivel educativo del padre', 'Admisión'),
    'nivel_educativo_madre': ('Nivel educativo de la madre', 'Admisión'),
    'ocupacion_padre': ('Ocupación del padre', 'Admisión'),
    'ocupacion_madre': ('Ocupación de la madre', 'Admisión'),
    'rango_ingreso_familiar': ('Rango de ingreso familiar mensual', 'Admisión'),
    'tipo_vivienda': ('Tenencia de la vivienda', 'Admisión'),
    'trabaja_actualmente': ('Indica si el postulante trabaja', 'Admisión'),
    'financia_estudios': ('Fuente de financiamiento de los estudios', 'Admisión'),
    'tipo_gestion_colegio': ('Tipo de gestión del colegio de procedencia', 'Admisión'),
    'area_colegio': ('Área urbana o rural del colegio de procedencia', 'Admisión'),
    'ugel_colegio': ('Unidad de Gestión Educativa Local del colegio', 'Admisión'),
    'programa_primera_opcion': ('Programa de estudios postulado', 'Admisión'),
    'facultad': ('Facultad del programa postulado', 'Admisión'),
    'area_programa': ('Área académica del programa postulado', 'Admisión'),
    'estudio_cepreuna_previo': ('Indica si estudió previamente en el CEPREUNA', 'Admisión'),
    'tipo_preparacion_preuniversitaria': ('Tipo de preparación preuniversitaria', 'Admisión'),
    'estudio_universidad_previamente': ('Indica si estudió antes en una universidad', 'Admisión'),
    'ie_turno': ('Turno de la institución educativa', 'SIAGIE'),
    'ie_forma_atencion_modelo_servicio': ('Forma de atención o modelo de servicio de la institución educativa', 'SIAGIE'),
    'ie_gestion_dependencia': ('Dependencia de gestión de la institución educativa', 'SIAGIE'),
}
for k, v in AREAS.items():
    DESC[k + '_media'] = (f'Promedio de {v} de tercero a quinto grado', 'Derivada del SIAGIE')

def fmt(x):
    return f'{x:.0f}' if float(x).is_integer() else f'{x:.2f}'

filas = [dict(variable='ingreso', descripcion='Variable dependiente: 1 si la condición final fue ingresó y 0 en otro caso',
              fuente='Derivada de Admisión', tipo='Binaria', valores='0 y 1')]
for c in NUM:
    v = pd.to_numeric(d[c], errors='coerce').dropna()
    filas.append(dict(variable=c, descripcion=DESC[c][0], fuente=DESC[c][1], tipo='Numérica',
                      valores=f'{fmt(v.min())} a {fmt(v.max())}'))
for c in CAT:
    k = d[c].dropna().astype(str).nunique()
    filas.append(dict(variable=c, descripcion=DESC[c][0], fuente=DESC[c][1], tipo='Categórica',
                      valores=f'{k} categorías'))
pd.DataFrame(filas).to_csv(S / 't_diccionario.csv', index=False)
print(pd.DataFrame(filas).to_string())
