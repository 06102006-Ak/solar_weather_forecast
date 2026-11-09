import numpy as np
import pandas as pd
from scipy.fft import rfft,rfftfreq

def fft_spectrum(values, sample_spacing=1.0):
    x=np.asarray(values,dtype=float); x=x[np.isfinite(x)]
    if len(x)<4: raise ValueError("Need at least four finite samples")
    x=x-x.mean()
    return pd.DataFrame({"frequency":rfftfreq(len(x),d=sample_spacing),"magnitude":np.abs(rfft(x))/len(x)})

def design(t,period,harmonics):
    t=np.asarray(t,dtype=float); cols=[np.ones_like(t)]
    for k in range(1,harmonics+1):
        a=2*np.pi*k*t/period; cols.extend([np.sin(a),np.cos(a)])
    return np.column_stack(cols)

class FourierRegressor:
    def __init__(self,period,harmonics=3):
        self.period=float(period); self.harmonics=int(harmonics); self.coef_=None; self.n_=0
    def fit(self,y):
        y=np.asarray(y,dtype=float)
        if not np.isfinite(y).all(): raise ValueError("Training values must be finite")
        t=np.arange(len(y)); self.coef_,*_=np.linalg.lstsq(design(t,self.period,self.harmonics),y,rcond=None); self.n_=len(y); return self
    def predict(self,steps):
        if self.coef_ is None: raise RuntimeError("fit first")
        t=np.arange(self.n_,self.n_+steps)
        return design(t,self.period,self.harmonics)@self.coef_
