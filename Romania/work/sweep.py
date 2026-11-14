import sys, numpy as np, pandas as pd, warnings, json
warnings.filterwarnings('ignore')
from scipy.stats import kurtosis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier as HGB, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
d=sys.argv[1]
tr=pd.read_csv(d+'/train.csv'); se=pd.read_csv(d+'/sealed.csv')
C=[f"c{j+1:02d}" for j in range(48)]
X=tr[C].values; y=tr.vrancea.values; Xs=se[C].values
Z=np.vstack([X,Xs]); k=kurtosis(X,axis=0); A=np.where(k>0.5)[0]; B=np.where(k<0.4)[0]
med=np.median(Z,0); g=np.sqrt(((Z[:,A]-med[A])**2).mean(1)); lowall=g<0.6
c13=Z[:,C.index('c13')]; net=np.where(lowall,0,np.where(c13<=-0.59,1,2)); nt=net[:7500]
cv=StratifiedKFold(5,shuffle=True,random_state=1)
sq=FunctionTransformer(lambda W: np.hstack([W,W**2]))
models={
 'lr':lambda cols: make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=5000)),
 'lr_sq':lambda cols: make_pipeline(StandardScaler(),sq,StandardScaler(),LogisticRegression(C=1,max_iter=5000)),
 'quad_lr':lambda cols: make_pipeline(StandardScaler(),PolynomialFeatures(2,include_bias=False),StandardScaler(),LogisticRegression(C=10,max_iter=5000)),
 'qda':lambda cols: QDA(reg_param=0.0),
 'svm_lin':lambda cols: make_pipeline(StandardScaler(),SVC(kernel='linear',C=0.1)),
 'svm_rbf':lambda cols: make_pipeline(StandardScaler(),SVC(C=3)),
 'svm_poly2':lambda cols: make_pipeline(StandardScaler(),SVC(kernel='poly',degree=2,C=1)),
 'mlp':lambda cols: make_pipeline(StandardScaler(),MLPClassifier((64,32),alpha=1e-3,max_iter=2000,early_stopping=True,random_state=0)),
 'hgb':lambda cols: HGB(max_iter=600,learning_rate=0.05),
 'rf':lambda cols: RandomForestClassifier(500,n_jobs=4,random_state=0),
}
feats={0:{'all':list(range(48)),'B':list(B)},1:{'all':list(range(48)),'k3':[C.index(c) for c in ['c35','c43','c38']]},2:{'all':list(range(48))}}
res={}
for i in range(3):
    m=nt==i
    for fn,cols in feats[i].items():
        for mn,mk in models.items():
            if mn=='quad_lr' and len(cols)>20: continue
            p=cross_val_predict(mk(cols),X[m][:,cols],y[m],cv=cv,n_jobs=5)
            a=(p==y[m]).mean(); res[f'{i}/{fn}/{mn}']=a; print(i,fn,mn,round(a,4),flush=True)
json.dump(res,open('sweep.json','w'),indent=1)
