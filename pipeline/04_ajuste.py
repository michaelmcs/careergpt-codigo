import os as _os
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import re, runpy, numpy as np, pandas as pd, itertools, json, warnings
warnings.filterwarnings('ignore')
src=open('pipeline/03_resultados.py').read()
cut=src.index("params = dict(")
ns={'__file__': __file__}; exec(src[:cut], ns)
X,y,g=ns['X'],ns['y'],ns['g']
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
cv=StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=7)
spw=(len(y)-y.sum())/y.sum()
grid=[dict(max_depth=md, learning_rate=lr, n_estimators=ne, min_child_weight=mc, colsample_bytree=cs, subsample=.8, reg_lambda=rl)
      for md,lr,ne,mc,cs,rl in [(2,.05,400,20,.6,5),(3,.03,500,20,.6,5),(3,.05,300,50,.5,10),(4,.03,400,30,.6,5),(2,.1,300,50,.5,10),(3,.02,800,40,.5,10),(6,.05,600,5,.8,1)]]
res=[]
for gp in grid:
    a=[]
    for tr,te in cv.split(X,y,g):
        m=XGBClassifier(**gp, scale_pos_weight=spw, enable_categorical=True, tree_method='hist', max_cat_to_onehot=1, random_state=42, n_jobs=-1).fit(X.iloc[tr],y[tr])
        a.append(roc_auc_score(y[te], m.predict_proba(X.iloc[te])[:,1]))
    res.append((np.mean(a),gp)); print(round(np.mean(a),4), gp, flush=True)
best=max(res,key=lambda r:r[0]); print('MEJOR',best)
json.dump(best[1], open('resultados/mejores_params.json','w'))
