import os as _os
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':10})
N={'promedio_5to':'Promedio de 5.° de secundaria','promedio_secundaria':'Promedio de secundaria','perfil_verbal':'Perfil verbal',
'equipamiento_tic':'Equipamiento TIC del hogar','tasa_previa':'Tasa previa de ingreso del programa','perfil_cuantitativo':'Perfil cuantitativo',
'tipo_preparacion_preuniversitaria':'Preparación preuniversitaria','dpcc_media':'Desarrollo Personal, Ciudadanía y Cívica',
'programa_primera_opcion':'Programa postulado','tipo_proceso':'Tipo de proceso de admisión','estudio_cepreuna_previo':'Estudió en CEPREUNA',
'ciencias_sociales_media':'Ciencias Sociales','matematica_media':'Matemática','comunicacion_media':'Comunicación','educacion_trabajo_media':'Educación para el Trabajo'}
imp=pd.read_csv('resultados/t_importancia.csv'); imp['etiqueta']=imp.variable.map(lambda v:N.get(v,v)); imp.to_csv('resultados/t_importancia.csv',index=False)
tipo={'Equipamiento TIC del hogar':'Condición socioeconómica','Preparación preuniversitaria':'Preparación preuniversitaria','Estudió en CEPREUNA':'Preparación preuniversitaria',
'Tasa previa de ingreso del programa':'Programa y proceso','Programa postulado':'Programa y proceso','Tipo de proceso de admisión':'Programa y proceso'}
col={'Rendimiento escolar':'#2E75B6','Condición socioeconómica':'#C55A11','Preparación preuniversitaria':'#548235','Programa y proceso':'#BF9000'}
top=imp.head(12).iloc[::-1]
fig,ax=plt.subplots(figsize=(7.2,4.4))
ax.barh(top.etiqueta,top.ganancia_rel,color=[col[tipo.get(e,'Rendimiento escolar')] for e in top.etiqueta],edgecolor='white')
for y_,v in enumerate(top.ganancia_rel): ax.text(v+.15,y_,f'{v:.1f}',va='center',fontsize=8)
ax.legend(handles=[Patch(color=c,label=k) for k,c in col.items()],loc='lower right',fontsize=8,frameon=False)
ax.set_xlabel('Ganancia relativa (%)'); ax.spines[['top','right']].set_visible(False); ax.set_xlim(0,top.ganancia_rel.max()*1.15)
plt.tight_layout(); plt.savefig('resultados/fig_importancia.png',dpi=220)
