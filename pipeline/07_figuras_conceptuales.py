import os as _os
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import pandas as pd
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':10})
AZ,NA,VE,DO,GR='#2E75B6','#C55A11','#548235','#BF9000','#7F7F7F'
def caja(ax,x,y,w,h,t,c,fs=9,tc='white'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.08',fc=c,ec='none'))
    ax.text(x+w/2,y+h/2,t,ha='center',va='center',color=tc,fontsize=fs,wrap=True)
def flecha(ax,x1,y1,x2,y2,c=GR):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=13,color=c,lw=1.4))

# ---- Figura LoRA
fig,ax=plt.subplots(figsize=(7.4,4.2)); ax.set_xlim(0,10); ax.set_ylim(0,6); ax.axis('off')
caja(ax,4.3,5.2,1.4,0.55,'Salida  h',GR)
ax.add_patch(Rectangle((1.2,1.6),3.0,2.6,fc='#DEEBF7',ec=AZ,lw=1.5))
ax.text(2.7,3.2,'Pesos preentrenados\n$W_0 \\in \\mathbb{R}^{d \\times k}$',ha='center',va='center',fontsize=10)
ax.text(2.7,2.0,'congelados',ha='center',fontsize=9,style='italic',color=AZ)
ax.add_patch(FancyBboxPatch((6.0,3.1),2.8,0.9,boxstyle='round,pad=0.02',fc='#FBE5D6',ec=NA,lw=1.5))
ax.text(7.4,3.55,'$B \\in \\mathbb{R}^{d \\times r}$  (inicia en 0)',ha='center',va='center',fontsize=9.5)
ax.add_patch(FancyBboxPatch((6.3,1.6),2.2,0.9,boxstyle='round,pad=0.02',fc='#FBE5D6',ec=NA,lw=1.5))
ax.text(7.4,2.05,'$A \\in \\mathbb{R}^{r \\times k}$  (aleatoria)',ha='center',va='center',fontsize=9.5)
ax.text(7.4,4.25,'entrenables, rango $r \\ll \\min(d,k)$',ha='center',fontsize=9,style='italic',color=NA)
caja(ax,4.3,0.2,1.4,0.55,'Entrada  x',GR)
flecha(ax,5.0,0.75,2.7,1.6); flecha(ax,5.0,0.75,7.4,1.6)
flecha(ax,7.4,2.5,7.4,3.1,NA); flecha(ax,2.7,4.2,4.8,5.2); flecha(ax,7.4,4.0,5.2,5.2,NA)
ax.text(5.0,4.6,'+',fontsize=18,ha='center',va='center',color=GR)
ax.text(8.0,4.85,'escala $\\alpha/r$',fontsize=9,color=NA)
ax.text(5.0,-0.25,'$h = W_0x + \\frac{\\alpha}{r}BAx$',ha='center',fontsize=11)
plt.tight_layout(); plt.savefig('resultados/fig_lora.png',dpi=220); plt.close()

# ---- Figura arquitectura CareER-GPT
fig,ax=plt.subplots(figsize=(7.6,4.6)); ax.set_xlim(0,12); ax.set_ylim(0,7.4); ax.axis('off')
caja(ax,0.2,5.6,2.4,1.0,'Base de admisión\nUNAP',GR); caja(ax,0.2,3.9,2.4,1.0,'Historial escolar\nSIAGIE',GR)
caja(ax,3.2,4.7,2.4,1.1,'Preprocesamiento\ny emparejamiento\npor identificador',AZ)
flecha(ax,2.6,6.1,3.2,5.4); flecha(ax,2.6,4.4,3.2,5.0)
ax.add_patch(Rectangle((6.1,3.6),5.7,3.5,fc='none',ec=AZ,ls='--',lw=1.2)); ax.text(8.95,6.85,'Componente tabular',ha='center',color=AZ,fontsize=9.5,weight='bold')
caja(ax,6.4,4.7,2.3,1.1,'Modelo XGBoost\nP(ingreso | perfil,\nprograma)',AZ)
caja(ax,9.2,4.7,2.3,1.1,'Calibración\nisotónica y\nrankeo',AZ)
flecha(ax,5.6,5.25,6.4,5.25); flecha(ax,8.7,5.25,9.2,5.25)
ax.add_patch(Rectangle((6.1,0.2),5.7,3.0,fc='none',ec=NA,ls='--',lw=1.2)); ax.text(8.95,0.35,'Componente lingüístico',ha='center',color=NA,fontsize=9.5,weight='bold')
caja(ax,9.2,1.9,2.3,1.0,'Instrucción:\nperfil + ranking',NA)
caja(ax,6.4,1.9,2.3,1.0,'LLM + adaptadores\nLoRA',NA)
caja(ax,3.2,1.9,2.4,1.0,'Ruta educativa\npersonalizada',VE)
flecha(ax,10.35,4.7,10.35,2.9); flecha(ax,9.2,2.4,8.7,2.4); flecha(ax,6.4,2.4,5.6,2.4)
caja(ax,0.2,1.9,2.4,1.0,'Postulante',DO); flecha(ax,3.2,2.4,2.6,2.4)
plt.tight_layout(); plt.savefig('resultados/fig_arquitectura.png',dpi=220); plt.close()

# ---- Figura proceso de entrenamiento con QLoRA
fig,ax=plt.subplots(figsize=(7.6,3.3)); ax.set_xlim(0,12.4); ax.set_ylim(0,4.4); ax.axis('off')
pasos=[('1. Corpus\nCareER-Dataset\n(JSONL)',GR),('2. Tokenización\ncon plantilla\nde chat',AZ),('3. Modelo base\ncuantizado\nen 4 bits (NF4)',AZ),
       ('4. Inserción de\nadaptadores LoRA\nr = 16, α = 32',NA),('5. Entrenamiento\nsupervisado\n(pérdida CE)',NA),('6. Evaluación y\nfusión de pesos',VE)]
for i,(t,c) in enumerate(pasos):
    x=0.1+i*2.07; caja(ax,x,1.9,1.85,1.6,t,c,fs=7.8)
    if i<5: flecha(ax,x+1.85,2.7,x+2.07,2.7)
ax.text(6.2,1.1,'Solo se actualizan A y B; los pesos del modelo base permanecen congelados.',ha='center',fontsize=9,style='italic')
ax.text(6.2,0.55,'La pérdida se monitorea en la partición de validación en cada época.',ha='center',fontsize=9,style='italic')
plt.tight_layout(); plt.savefig('resultados/fig_entrenamiento.png',dpi=220); plt.close()

# ---- Figura rankeo
T=pd.read_csv('resultados/t_ranking.csv'); met=['MRR','NDCG5','NDCG10','Hit1','Hit3']; lab=['MRR','NDCG@5','NDCG@10','Acierto@1','Acierto@3']
fig,ax=plt.subplots(figsize=(7.2,3.8)); import numpy as np; x=np.arange(len(met)); w=0.26
for i,(r,c) in enumerate(zip(T.itertuples(),[GR,VE,AZ])):
    v=[getattr(r,m) for m in met]; b=ax.bar(x+(i-1)*w,v,w,color=c,label=r.metodo)
    for xi,vi in zip(x+(i-1)*w,v): ax.text(xi,vi+.008,f'{vi:.2f}',ha='center',fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylabel('Valor'); ax.legend(frameon=False,fontsize=8.5); ax.spines[['top','right']].set_visible(False)
ax.set_ylim(0,max(T[met].max())*1.18); plt.tight_layout(); plt.savefig('resultados/fig_ranking.png',dpi=220); plt.close()
print('figuras OK')
