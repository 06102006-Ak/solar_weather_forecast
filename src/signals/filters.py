import numpy as np
from scipy.signal import lfilter,freqz

def causal_moving_average(values,window=3):
    if window<1: raise ValueError("window must be >= 1")
    return lfilter(np.ones(window)/window,[1.0],np.asarray(values,dtype=float))

def moving_average_frequency_response(window=3,worN=512):
    if window<1: raise ValueError("window must be >= 1")
    return freqz(np.ones(window)/window,[1.0],worN=worN)
