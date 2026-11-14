import sys, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.mixture import GaussianMixture
d=sys.argv[1]
tr=pd.read_csv(d+'/train.csv'); se=pd.read_csv(d+'/sealed.csv')
C=[f"c{j+1:02d}" for j in range(48)]
X=tr[C].values; y=tr.vrancea.values; Xs=se[C].values
print(tr.describe().T[['mean','std','min','max']].round(2).to_string())
print('sealed mean/std', Xs.mean(0).round(2), Xs.std(0).round(2))
print('nan', np.isnan(X).sum(), 'dup ids', tr.record_id.duplicated().sum(), 'overlap', len(set(tr.record_id)&set(se.record_id)))
print('corr with y', np.round([np.corrcoef(X[:,j],y)[0,1] for j in range(48)],2))
cv=StratifiedKFold(5,shuffle=True,random_state=0)
for name,m in {'lr':make_pipeline(StandardScaler(),LogisticRegression(max_iter=5000)),
 'svm':make_pipeline(StandardScaler(),SVC(C=3)),
 'rf':RandomForestClassifier(500,n_jobs=-1),
 'hgb':HistGradientBoostingClassifier(),
 'knn':make_pipeline(StandardScaler(),KNeighborsClassifier(15)),
 'mlp':make_pipeline(StandardScaler(),MLPClassifier((128,64),max_iter=500,early_stopping=True))}.items():
    print(name, cross_val_score(m,X,y,cv=cv,n_jobs=5).mean().round(4), flush=True)
Z=StandardScaler().fit_transform(np.vstack([X,Xs]))
for k in range(1,7):
    g=GaussianMixture(k,n_init=3,random_state=0).fit(Z); print('gmm',k,round(g.bic(Z)))
