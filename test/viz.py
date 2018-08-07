import sys
import os
sys.path.append(os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import LinearNDInterpolator
from scipy import optimize

from vessel_tracking import geometry, util
from tqdm import tqdm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-input_dir')
parser.add_argument('--path_dir', type=str, default='./vessel_paths')
args   = parser.parse_args()

#############################################
# Import image and points
#############################################
im_file = [a for a in os.listdir(args.input_dir) if ".mha" in a][0]
I            = util.load_image(args.input_dir+'/'+im_file)
meta         = util.load_json(args.input_dir+'/meta.json')
points_j       = util.load_json(args.input_dir+'/points.json')
start_points_j = util.load_json(args.input_dir+'/start_points.json')

points = np.array(points_j['points'])
##############################################

##############################################
# Import Vessels
##############################################
vessel_dir = os.path.abspath(args.path_dir)
folders = os.listdir(vessel_dir)
if len(folders) == 0:
    raise RuntimeError("no vessel path folders found in {}".format(vessel_dir))

paths = []
for f in folders:
    vdir = vessel_dir+'/'+f
    path_points = util.load_json(vdir+'/path_points.json')['path_points']

    paths.append(np.array(path_points))

##############################################
N = int(meta['HEIGHT']/meta['SPACING'])
x = np.linspace(-meta['HEIGHT'],meta['HEIGHT'],N)

X,Y = np.meshgrid(x,x)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.contourf(Y,X,I[:,:,0], zdir='z', offset=0, cmap=cm.gray)

for path in paths:
    ax.plot(path[:,0],path[:,1],path[:,2],color='g',marker='o')

ax.set_xlim(-meta['HEIGHT'],meta['HEIGHT'])
ax.set_ylim(-meta['HEIGHT'],meta['HEIGHT'])
ax.set_zlim(-meta['HEIGHT'],meta['HEIGHT'])
plt.show()
plt.close()
