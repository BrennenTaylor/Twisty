import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import signal
import math

def CosFunction(t):
    return np.cos(np.pi * t / 2)

def GaussianPhase(N, w, wp, mu):
    dot = np.dot(w, wp)
    return np.exp(-1 * (1 - dot) / mu)

step = 0.01
x = np.arange(-3, 3, step)
numValues = x.size
print(numValues)
print("\n")
y = CosFunction(x)
window = scipy.signal.boxcar((int)(2 / step))
windowPadSize = (numValues - window.size) / 2
print(windowPadSize)
print("\n")
window = np.lib.pad(window, (int(windowPadSize), int(windowPadSize)), 'constant')
y = y * window
y_fft = np.fft.fftshift(np.abs(scipy.fft.fft(y))) / np.sqrt(len(y))
plt.plot(x, y)
plt.plot(x,y_fft)
plt.show()