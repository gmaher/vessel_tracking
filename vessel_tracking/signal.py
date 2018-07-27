import numpy as np

def smooth(x):
    n = x.shape[0]

    y = np.zeros((n))

    y[0]   = x[0]
    y[n-1] = x[n-1]

    y[1:n-1] = (x[0:n-2]+x[1:n-1]+x[2:n])*(1.0/3)

    return y

def smooth_n(x,N=2):
    y = x.copy()

    for i in range(N):
        y = smooth(y)

    return y

def central_difference(x, dx=1):
    n = x.shape[0]

    d = np.zeros((n))

    d[0] = (x[1]-x[0])/dx
    d[n-1] = (x[-1]-x[-2])/dx

    d[1:n-1] = (x[2:n]-x[:n-2])/(2*dx)

    return d

def find_peaks(y, n_smooth=0, tol=0.05, dx=1):
    z = y.copy()

    z = smooth_n(z,n_smooth)

    dz = central_difference(z,dx)

    peak_or_not = np.abs(dz) <= tol

    inds = np.arange(len(z))

    peak_inds = inds[peak_or_not]

    return peak_inds
