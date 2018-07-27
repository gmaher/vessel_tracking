import sys
import os
sys.path.append(os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import LinearNDInterpolator

from vessel_tracking import geometry, signal

fn = './data/aorta.npy'
image = np.load(fn)

W,H = image.shape

x = np.linspace(-1,1,W)

X,Y = np.meshgrid(x,x)

Xv = np.ravel(X)
Yv = np.ravel(Y)
points = np.concatenate((Xv[:,np.newaxis], Yv[:,np.newaxis]), axis=1)
I  = np.ravel(image)

I_int = LinearNDInterpolator(points, I)

N_rays    = 20
steps     = 50
step_size = 0.01
peak_factor = 0.5

origin = np.array([0,0])
angles = np.linspace(0,2*np.pi,N_rays)

plt.figure()
plt.imshow(image, extent=[-1,1,1,-1], cmap='gray')
plt.colorbar()

ray_intensities = []
peaks = []
cuts = []
cuts2 = []

for i in range(N_rays):
    theta = angles[i]
    d     = np.array([np.cos(theta), np.sin(theta)])

    ray = geometry.ray(origin,d,step_size,steps,False)

    intensities = I_int(ray)

    z = signal.smooth_n(intensities,4)
    dz = signal.central_difference(z)

    s_ind = np.argmin(dz)
    dz_min = np.amin(dz)

    peaks.append(ray[s_ind])

    ray_intensities.append(intensities)

    plt.scatter(ray[:,0], ray[:,1])

plt.show()
plt.close()

peaks = np.array(peaks)

plt.figure()
plt.imshow(image, extent=[-1,1,1,-1], cmap='gray')
plt.colorbar()

plt.scatter(peaks[:,0],peaks[:,1], color='r', marker="o")

plt.show()
plt.close()

# plt.figure()
# for r in ray_intensities:
#     plt.plot(r)
# plt.show()
# plt.close()
#
plt.figure()
for r in ray_intensities:
    z = signal.smooth_n(r,3)
    dz = signal.central_difference(z)
    dz2 = signal.central_difference(dz)

    plt.plot(dz)
plt.show()
plt.close()
