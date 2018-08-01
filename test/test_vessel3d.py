import sys
import os
sys.path.append(os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import LinearNDInterpolator

from vessel_tracking import geometry, signal, sample, ransac

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def vessel(x,r):
    if np.sqrt(x[0]**2+x[1]**2)<=r:
        return 1
    else:
        return 0

Nr = 100
Np = 50
c0 = np.array([0,0,0])
r0 = 1.0
dt = np.array([0,0.1,0.9])
dt = dt/np.sqrt(np.sum(dt**2))

N = 40
HEIGHT  = 5
WIDTH   = 5
DEPTH   = 5
SPACING = HEIGHT*1.0/N

RADIUS  = 1.5

x = np.linspace(-HEIGHT,HEIGHT,N)

X,Y = np.meshgrid(x,x)

I = np.zeros((N,N,N))
points = []

for i in range(N):
    for j in range(N):
        for k in range(N):
            v = np.array([x[i], x[j], x[k]])

            points.append(v.copy())

            I[i,j,k] = vessel(v, RADIUS)

points = np.array(points)

I_int = LinearNDInterpolator(points, np.ravel(I))

#ray casting
step_size  = np.sqrt(10*RADIUS)/Np

surface_points = ransac.sample_surface_points(I_int, c0, step_size, Nr, Np)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(surface_points[:,0], surface_points[:,1],
    surface_points[:,2], color='b')

ax.contourf(X,Y,I[:,:,0], zdir='z', offest=0, cmap=cm.gray)
ax.set_xlim(-HEIGHT,HEIGHT)
ax.set_ylim(-HEIGHT,HEIGHT)
ax.set_zlim(-HEIGHT,HEIGHT)
plt.show()
plt.close()
