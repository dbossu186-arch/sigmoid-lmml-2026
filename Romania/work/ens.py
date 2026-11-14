import sys, numpy as np, pandas as pd, warnings, json
warnings.filterwarnings('ignore')
from scipy.stats import kurtosis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier as HGB, RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.model_selection import StratifiedKFold
d=sys.argv[1]
tr=pd.read_csv(d+'/train.csv'); se=pd.read_csv(d+'/sealed.csv')
C=[f"c{j+1:02d}" for j in range(48)]
X=tr[C].values; y=tr.vrancea.values; Xs=se[C].values
Z=np.vstack([X,Xs]); k=kurtosis(X,axis=0); A=np.where(k>0.5)[0]; B=list(np.where(k<0.4)[0])
med=np.median(Z,0); g=np.sqrt(((Z[:,A]-med[A])**2).mean(1)); lowall=g<0.6
c13=Z[:,C.index('c13')]; net=np.where(lowall,0,np.where(c13<=-0.59,1,2)); nt=net[:7500]; ns=net[7500:]
K3=[C.index(c) for c in ['c35','c43','c38']]; ALL=list(range(48))
M={0:{'quad_lr':(B,lambda: make_pipeline(StandardScaler(),PolynomialFeatures(2,include_bias=False),StandardScaler(),LogisticRegression(C=1,max_iter=20000))),
      'svm_rbf':(B,lambda: make_pipeline(StandardScaler(),SVC(C=3,probability=True,random_state=0))),
      'qda':(B,lambda: QDA())},
   1:{'hgb':(ALL,lambda: HGB(max_iter=600,learning_rate=0.05)),
      'rf':(K3,lambda: RandomForestClassifier(500,min_samples_leaf=3,n_jobs=4,random_state=0)),
      'et':(K3,lambda: ExtraTreesClassifier(500,min_samples_leaf=3,n_jobs=4,random_state=0))},
   2:{'lr':(ALL,lambda: make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=5000))),
      'svm_lin':(ALL,lambda: make_pipeline(StandardScaler(),SVC(kernel='linear',C=0.1,probability=True,random_state=0))),
      'lr_c03':(ALL,lambda: make_pipeline(StandardScaler(),LogisticRegression(C=0.3,max_iter=5000)))}}
res={}; allp={}
for rep in range(3):
  cv=StratifiedKFold(5,shuffle=True,random_state=rep)
  for i in range(3):
    idx=np.where(nt==i)[0]; P={n:np.zeros(len(idx)) for n in M[i]}; PST={n:np.zeros(len(idx)) for n in M[i]}
    for a,b in cv.split(idx,y[idx]):
      for n,(cols,mk) in M[i].items():
        m=mk().fit(X[idx[a]][:,cols],y[idx[a]]); P[n][b]=m.predict_proba(X[idx[b]][:,cols])[:,1]
        # self-training: add confident sealed pseudo-labels of same network
        S=Xs[ns==i][:,cols]; ps=m.predict_proba(S)[:,1]; conf=(ps>0.9)|(ps<0.1)
        m2=mk().fit(np.vstack([X[idx[a]][:,cols],S[conf]]),np.r_[y[idx[a]],(ps[conf]>0.5).astype(int)]); PST[n][b]=m2.predict_proba(X[idx[b]][:,cols])[:,1]
    yy=y[idx]
    for n in M[i]:
      res.setdefault(f'{i}/{n}',[]).append(((P[n]>0.5)==yy).mean()); res.setdefault(f'{i}/{n}+self',[]).append(((PST[n]>0.5)==yy).mean())
    avg=np.mean([P[n] for n in M[i]],0); res.setdefault(f'{i}/ENS',[]).append(((avg>0.5)==yy).mean())
    avg=np.mean([PST[n] for n in M[i]],0); res.setdefault(f'{i}/ENS+self',[]).append(((avg>0.5)==yy).mean())
  print('rep',rep,flush=True)
for kk,v in res.items(): print(kk,np.round(np.mean(v),4),np.round(v,4))
json.dump({k:float(np.mean(v)) for k,v in res.items()},open('ens.json','w'),indent=1)
