import sys
import os
sys.path.append(os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import LinearNDInterpolator

from vessel_tracking import geometry, signal

fn = './data/rca.npy'
image = np.load(fn)

W,H = image.shape

x = np.linspace(-1,1,W)

X,Y = np.meshgrid(x,x)

Xv = np.ravel(X)
Yv = np.ravel(Y)
points = np.concatenate((Xv[:,np.newaxis], Yv[:,np.newaxis]), axis=1)
I  = np.ravel(image)

I_int = LinearNDInterpolator(points, I)

N_rays    = 10
steps     = 10
step_size = 0.02

origin = np.array([0,0])
angles = np.linspace(-2*np.pi,2*np.pi,N_rays)

plt.figure()
plt.imshow(image, extent=[-1,1,1,-1], cmap='gray')
plt.colorbar()

ray_intensities = []

for i in range(N_rays):
    theta = angles[i]
    d     = np.array([np.cos(theta), np.sin(theta)])

    ray = geometry.ray(origin,d,step_size,steps)

    intensities = I_int(ray)

    ray_intensities.append(intensities)

    plt.scatter(ray[:,0], ray[:,1])

plt.show()
plt.close()

plt.figure()
for r in ray_intensities:
    plt.plot(r)
plt.show()
plt.close()
