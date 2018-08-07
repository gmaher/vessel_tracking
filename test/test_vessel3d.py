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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-input_dir')
parser.add_argument('--out_dir', type=str, default='.')
args   = parser.parse_args()

Nr = 50
Np = 30
Nd = 50
Ncylinder     = 300
p_in          = 0.65
max_dev       = 0.5
inlier_factor = 0.15
step_size     = 4

#############################################
# Import image and points
#############################################
im_file = [a for a in os.listdir(args.input_dir) if ".mha" in a][0]
I            = util.load_image(args.input_dir+'/'+im_file)
meta         = util.load_json(args.input_dir+'/meta.json')
points_j       = util.load_json(args.input_dir+'/points.json')
start_points_j = util.load_json(args.input_dir+'/start_points.json')

points = np.array(points_j['points'])

#############################################
#############################################

print("interpolating image")
I_int = LinearNDInterpolator(points, np.ravel(I))

tracker = algorithm.RansacVesselTracker()
tracker.set_params(Nr=Nr, Np=Np, Nd=Nd, Nc=Ncylinder, p_in=p_in, max_dev=max_dev,
    inlier_factor=inlier_factor, step_size=step_size)

tracker.set_image(I_int)

print("starting tracker")

output_paths = {}

cur_dir = os.path.abspath(args.out_dir+'/vessel_paths')
try:
    os.mkdir(cur_dir)
except:
    print(cur_dir, " already exists")

for start in start_points_j:
    sp = start_points_j[start]

    c0 = np.array(sp['x'])
    d0 = np.array(sp['d'])
    r0 = sp['r']
    h0 = sp['h']

    cylinders = tracker.get_path(d0,c0,r0,h0)

    p_dir = cur_dir+'/'+start
    try:
        os.mkdir(p_dir)
    except:
        print(p_dir, " already exists")

    path_points = [c[1].tolist() for c in cylinders]

    surf_points = []
    for C in cylinders:
        if len(C[5]) > 0:
            surf_points = surf_points + [C[5]]
    surf_points = np.concatenate(surf_points,axis=0).tolist()

    p_dict = {"path_points":path_points}
    util.write_json(p_dict, p_dir+'/path_points.json')
    s_dict = {"surface_points":surf_points}
    util.write_json(s_dict, p_dir+'/surface_points.json')
