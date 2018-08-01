import sys
import os
sys.path.append(os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import LinearNDInterpolator

from vessel_tracking import geometry, signal, sample, ransac
from tqdm import tqdm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def vessel(x,r):
    if np.sqrt(x[0]**2+x[1]**2)<=r:
        return 1
    else:
        return 0

Nr = 200
Np = 50
Nd = 50
Ncylinder     = 200
p_in          = 0.7
max_dev       = 0.5
inlier_factor = 0.15

c0 = np.array([0,0,0])
r0 = 1.2
dt = np.array([0,0.1,0.9])
dt = dt/np.sqrt(np.sum(dt**2))

N       = 60
HEIGHT  = 5
WIDTH   = 5
DEPTH   = 5
SPACING = HEIGHT*1.0/N

RADIUS  = 1.5
NOISE_FACTOR = 0.2

x = np.linspace(-HEIGHT,HEIGHT,N)

X,Y = np.meshgrid(x,x)

I = np.zeros((N,N,N))
NOISE = np.random.randn(N,N,N)*NOISE_FACTOR
points = []

for i in range(N):
    for j in range(N):
        for k in range(N):
            v = np.array([x[i], x[j], x[k]])

            points.append(v.copy())

            I[i,j,k] = vessel(v, RADIUS) + NOISE[i,j,k]

points = np.array(points)

I_int = LinearNDInterpolator(points, np.ravel(I))

#ray casting
step_size  = np.sqrt(3*RADIUS)/Np

surface_points = ransac.sample_surface_points(I_int, c0, step_size, Nr, Np)

#get cylinder
candidate_axes = sample.sphere_sample_vec(dt,Nd)
curr_best_p = 0

curr_best_d = 0
curr_best_c = 0
curr_best_r = 0
curr_best_d = 0
curr_in = 0
curr_out = 0

for i in range(Nd):
    dtest = candidate_axes[i]
    print(dtest, dt, np.sum(dt*dtest))

    best_center, best_r, best_in_rate, inliers, outliers = \
        ransac.ransac_cylinder(surface_points, c0, dtest, r0, p_in,
            max_dev, inlier_factor, Ncylinder)

    if best_in_rate > curr_best_p:
        curr_best_p = best_in_rate
        curr_best_d = dtest
        curr_best_c = best_center
        curr_best_r = best_r
        curr_in     = inliers
        curr_out    = outliers

    if best_in_rate > p_in and ( (best_r <= 1.5*r0) and (best_r > 0.5*r0) ):
        print("acceptable cylinder found {}, {}, {}, {}"\
            .format(curr_best_d, curr_best_c, curr_best_r, curr_best_p))

        break

Ps = geometry.cylinder_surface(-1,1,20, curr_best_c, curr_best_d, curr_best_r)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# ax.scatter(surface_points[:,0], surface_points[:,1],
#     surface_points[:,2], color='b')

ax.contourf(X,Y,I[:,:,0], zdir='z', offest=0, cmap=cm.gray)

ax.plot_surface(Ps[:,:,0], Ps[:,:,1], Ps[:,:,2])

ax.scatter(curr_in[:,0],curr_in[:,1],curr_in[:,2],color='g')
ax.scatter(curr_out[:,0],curr_out[:,1],curr_out[:,2],color='r')

ax.set_xlim(-HEIGHT,HEIGHT)
ax.set_ylim(-HEIGHT,HEIGHT)
ax.set_zlim(-HEIGHT,HEIGHT)
plt.show()
plt.close()
