import sys, numpy as np, pandas as pd
from scipy.stats import kurtosis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
d=sys.argv[1]
tr=pd.read_csv(d+'/train.csv'); se=pd.read_csv(d+'/sealed.csv')
C=[f"c{j+1:02d}" for j in range(48)]
X=tr[C].values; y=tr.vrancea.values; Xs=se[C].values
Z=np.vstack([X,Xs]); k=kurtosis(X,axis=0); A=np.where(k>0.5)[0]; B=np.where(k<0.4)[0]
mu=np.median(Z[:,A],0); ZA=Z[:,A]-mu; g=np.sqrt((ZA**2).mean(1)); low=g<0.6
print('low frac train',low[:7500].mean(),'sealed',low[7500:].mean(),'y|low',y[low[:7500]].mean(),'y|high',y[~low[:7500]].mean())
# per-cluster scale
s_hi=ZA[~low].std(0); s_lo=ZA[low].std(0); print('ratio per channel',np.round(s_lo/s_hi,3))
# B channels per cluster
print('B mean hi',np.round(Z[~low][:,B].mean(0),2)); print('B mean lo',np.round(Z[low][:,B].mean(0),2))
print('B std hi',np.round(Z[~low][:,B].std(0),2)); print('B std lo',np.round(Z[low][:,B].std(0),2))
N=Z.copy(); N[:,A]=ZA/np.where(low,s_lo.mean()/s_hi.mean(),1)[:,None]
np.save('norm.npy',N); np.save('low.npy',low)
cv=StratifiedKFold(5,shuffle=True,random_state=0)
Xn=N[:7500]
for nm,m in [('hgb',HistGradientBoostingClassifier(max_iter=500,learning_rate=0.05)),('svm',make_pipeline(StandardScaler(),SVC(C=3))),('lr',make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000)))]:
    print(nm,'norm all',cross_val_score(m,Xn,y,cv=cv).mean().round(4),'normA only',cross_val_score(m,Xn[:,A],y,cv=cv).mean().round(4),flush=True)
# GMM within high cluster on normalized
H=N[~low]
for kk in [1,2,3]:
    gm=GaussianMixture(kk,n_init=2,random_state=0).fit(H); print('hi gmm',kk,round(gm.bic(H)),np.bincount(gm.predict(H)))
L=N[low]
for kk in [1,2,3]:
    gm=GaussianMixture(kk,n_init=2,random_state=0).fit(L); print('lo gmm',kk,round(gm.bic(L)),np.bincount(gm.predict(L)))
