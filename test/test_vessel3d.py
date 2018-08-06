import sys
import os
sys.path.append(os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import LinearNDInterpolator
from scipy import optimize

from vessel_tracking import geometry, signal, sample, ransac, algorithm, util
from tqdm import tqdm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import importlib

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-image')
parser.add_argument('-meta')
parser.add_argument('-points')
parser.add_argument('-start_points')
parser.add_argument('--out_dir', type=str, default='.')
args   = parser.parse_args()

Nr = 50
Np = 30
Nd = 50
Ncylinder     = 300
p_in          = 0.65
max_dev       = 0.5
inlier_factor = 0.15
step_size     = 6

#############################################
# Import image and points
#############################################
I            = util.load_image(args.image)
meta         = util.load_json(args.meta)
points_j       = util.load_json(args.points)
start_points_j = util.load_json(args.start_points)

points = np.array(points_j['points'])

c0 = np.array(start_points_j['start_points'][0]['x'])
d0 = np.array(start_points_j['start_points'][0]['d'])
r0 = start_points_j['start_points'][0]['r']
h0 = start_points_j['start_points'][0]['h']
#############################################
#############################################

print("interpolating image")
I_int = LinearNDInterpolator(points, np.ravel(I))

tracker = algorithm.RansacVesselTracker()
tracker.set_params(Nr=Nr, Np=Np, Nd=Nd, Nc=Ncylinder, p_in=p_in, max_dev=max_dev,
    inlier_factor=inlier_factor, step_size=step_size)

tracker.set_image(I_int)

print("starting tracker")

cylinders = tracker.get_path(d0,c0,r0,h0)

#################
N = int(meta['HEIGHT']/meta['SPACING'])
x = np.linspace(-meta['HEIGHT'],meta['HEIGHT'],N)

X,Y = np.meshgrid(x,x)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.contourf(X,Y,I[:,:,0], zdir='z', offset=0, cmap=cm.gray)

for C in cylinders:
    d = C[0]
    c = C[1]
    r = C[2]
    h = C[3]
    p = C[4]
    in_ = C[5]

    Ps = geometry.cylinder_surface(-h/2, h/2, 5, c,d,r)

    ax.plot_surface(Ps[:,:,0], Ps[:,:,1], Ps[:,:,2], color='b')

ax.set_xlim(-meta['HEIGHT'],meta['HEIGHT'])
ax.set_ylim(-meta['HEIGHT'],meta['HEIGHT'])
ax.set_zlim(-meta['HEIGHT'],meta['HEIGHT'])
plt.show()
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.contourf(X,Y,I[:,:,0], zdir='z', offset=0, cmap=cm.gray)

for C in cylinders:
    d = C[0]
    c = C[1]
    r = C[2]
    h = C[3]
    p = C[4]
    in_ = C[5]

    if len(in_)>0:
        ax.scatter(in_[:,0],in_[:,1],in_[:,2],color='g')
#
# ax.scatter(path_points[:,0],path_points[:,1],path_points[:,2],
#     color='k')
#
ax.set_xlim(-meta['HEIGHT'],meta['HEIGHT'])
ax.set_ylim(-meta['HEIGHT'],meta['HEIGHT'])
ax.set_zlim(-meta['HEIGHT'],meta['HEIGHT'])
plt.show()
plt.close()
