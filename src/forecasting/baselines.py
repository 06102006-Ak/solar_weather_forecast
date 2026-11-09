import numpy as np
from src.signals.fourier import FourierRegressor

def persistence(history,horizon):
    x=np.asarray(history,dtype=float)
    if not len(x): raise ValueError("empty history")
    return np.repeat(x[-1],horizon)

def seasonal_naive(history,horizon,season):
    x=np.asarray(history,dtype=float)
    if season<1 or len(x)<season: raise ValueError("not enough history for season")
    return np.tile(x[-season:],int(np.ceil(horizon/season)))[:horizon]

def fourier(history,horizon,period,harmonics=3):
    return FourierRegressor(period,harmonics).fit(history).predict(horizon)
