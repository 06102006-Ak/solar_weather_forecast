import numpy as np

def regression_metrics(y_true,y_pred):
    a=np.asarray(y_true,dtype=float); p=np.asarray(y_pred,dtype=float)
    if a.shape!=p.shape: raise ValueError("shape mismatch")
    mask=np.isfinite(a)&np.isfinite(p); a,p=a[mask],p[mask]
    if not len(a): raise ValueError("no finite paired observations")
    e=p-a
    return {"n":int(len(a)),"mae":float(np.mean(np.abs(e))),"rmse":float(np.sqrt(np.mean(e**2)))}
