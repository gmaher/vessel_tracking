import sys
import os
sys.path.append(os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt

from vessel_tracking import signal

N = 100
periods = 3
B = periods*2*np.pi
noise = 0.3
N_smooth = 4
tol = 0.05
dx = 1

x = np.linspace(-B,B,N)

y = np.sin(x)+1 +np.random.randn(N)*noise
y_smooth = signal.smooth_n(y, N=N_smooth)

dy = signal.central_difference(y)
dy_smooth = signal.central_difference(y_smooth)

peaks = signal.find_peaks(y,N_smooth,tol,dx)

plt.figure()
plt.plot(x,y,  color='b', label='y')
plt.plot(x,y_smooth,  color='g', label='y_smooth')
plt.plot(x,dy, color='r', label='dy')
plt.plot(x,dy_smooth, color='k', label='dy')
plt.scatter(x[peaks],y[peaks], color='r', marker='*')
plt.scatter(x[peaks],dy_smooth[peaks], color='r', marker='*')
plt.legend()
plt.show()
plt.close()
