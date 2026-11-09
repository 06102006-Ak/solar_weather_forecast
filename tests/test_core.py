import numpy as np
from src.signals.filters import causal_moving_average
from src.signals.fourier import FourierRegressor,fft_spectrum
from src.forecasting.baselines import persistence,seasonal_naive
from src.evaluation.metrics import regression_metrics

def test_persistence(): assert np.allclose(persistence([1,2,3],3),[3,3,3])
def test_seasonal(): assert np.allclose(seasonal_naive([1,2,3,4],6,2),[3,4,3,4,3,4])
def test_causal_filter(): assert np.allclose(causal_moving_average([0,0,9,0],2),[0,0,4.5,4.5])
def test_fourier():
    t=np.arange(100); y=3+2*np.sin(2*np.pi*t/24)
    assert len(FourierRegressor(24,2).fit(y).predict(24))==24
def test_fft_period():
    t=np.arange(240); spec=fft_spectrum(np.sin(2*np.pi*t/24))
    idx=spec.magnitude.iloc[1:].idxmax()
    assert abs(spec.frequency.iloc[idx]-1/24)<1e-6
def test_metrics(): assert regression_metrics([1,2,3],[1,2,4])["mae"]==1/3
