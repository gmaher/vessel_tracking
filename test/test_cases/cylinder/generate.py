import numpy as np
import os
import sys
import SimpleITK as sitk

sys.path.append(os.path.abspath('../../..'))

from vessel_tracking import util

def vessel(x,r):
    if np.sqrt(x[0]**2+x[1]**2)<=r:
        return 1
    else:
        return 0

c0 = np.array([0.5,0.5,-3])
r0 = 1.2
dt = np.array([0,0.1,0.9])
dt = dt/np.sqrt(np.sum(dt**2))
h0 = 0.5


N       = 40
HEIGHT  = 5
WIDTH   = 5
DEPTH   = 5
SPACING = HEIGHT*1.0/N

RADIUS  = 1.5
NOISE_FACTOR = 0.3

x = np.linspace(-HEIGHT,HEIGHT,N)

X,Y = np.meshgrid(x,x)

I = np.zeros((N,N,N))
NOISE = np.random.randn(N,N,N)*NOISE_FACTOR
points = []

print("Setting up image")
for i in range(N):
    for j in range(N):
        for k in range(N):
            v = np.array([x[i], x[j], x[k]])

            points.append(v.copy())

            I[i,j,k] = vessel(v, RADIUS) + NOISE[i,j,k]

points = np.array(points).reshape((-1,3)).tolist()
p = {"points":points}
util.write_json(p, 'points.json')

im = sitk.GetImageFromArray(I)
sitk.WriteImage(im,'cylinder.mha')

meta = {}
meta['HEIGHT'] = HEIGHT
meta['WIDTH'] = WIDTH
meta['DEPTH'] = DEPTH
meta['SPACING'] = SPACING
meta['DIMENSIONS'] = [N,N,N]
meta['EXTENT'] = [-HEIGHT,HEIGHT,-HEIGHT,HEIGHT,-HEIGHT,HEIGHT]

util.write_json(meta, 'meta.json')

start_points = {}
start_points['test_1'] = \
{
"x":c0.tolist(),
"d":dt.tolist(),
"r":r0,
"h":h0
}

util.write_json(start_points,'start_points.json')
