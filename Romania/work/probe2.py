import sys, numpy as np, pandas as pd
from scipy.stats import kurtosis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier
d=sys.argv[1]
tr=pd.read_csv(d+'/train.csv'); se=pd.read_csv(d+'/sealed.csv')
C=[f"c{j+1:02d}" for j in range(48)]
X=tr[C].values; y=tr.vrancea.values; Xs=se[C].values
k=kurtosis(X,axis=0); A=np.where(k>0.5)[0]; B=np.where(k<0.4)[0]
XA=np.vstack([X,Xs])[:,A]; Z=(XA-XA.mean(0))
ev,V=np.linalg.eigh(np.cov(Z.T)); print('eig A',np.round(ev[::-1],3))
cv=StratifiedKFold(5,shuffle=True,random_state=0)
h=lambda: HistGradientBoostingClassifier(max_iter=500,learning_rate=0.05)
print('hgb A only',cross_val_score(h(),X[:,A],y,cv=cv,n_jobs=5).mean())
print('hgb B only',cross_val_score(h(),X[:,B],y,cv=cv,n_jobs=5).mean())
# projections on top PCs
P=Z@V[:,::-1]
for i in range(12):
    p=P[:7500,i]; print('PC',i,'kurt',round(kurtosis(P[:,i]),2),'corr y',round(np.corrcoef(p,y)[0,1],3), 'hist',np.histogram(P[:,i],12)[0])
