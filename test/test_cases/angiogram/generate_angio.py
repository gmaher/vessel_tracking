import numpy as np
import os
import sys
import SimpleITK as sitk
import matplotlib.pyplot as plt
from queue import Queue
from scipy import ndimage

sys.path.append(os.path.abspath('../../..'))

from vessel_tracking import util, rl
from vessel_tracking.graph import Graph, DetectTree

class CollisionDetector(object):
    def __init__(self,I_int,n=100):
        self.I_int = I_int
        self.n = n

    def collision(self,p1,p2):
        for i in np.linspace(0,1,self.n):
            p = (1-i)*p1+i*p2
            if float(self.I_int(p)) >= THRESH:
                return True
        return False

def cost(p,pp):
    return np.linalg.norm(p-pp)


FILE = 'angio_1.jpg'
THRESH = 140

I = ndimage.imread(FILE, flatten=True)
I = ndimage.filters.gaussian_filter(I,sigma=2)

S = I.shape

H = S[0]
W = S[1]

points = np.zeros((H*W,2))
for i in range(H):
    for j in range(W):
        points[j+i*W,0] = i
        points[j+i*W,1] = j


from scipy.interpolate import LinearNDInterpolator

Im_int = LinearNDInterpolator(points+0.5, np.ravel(I))

Nsamp = 3000
p_samp = np.random.rand(Nsamp,2)
p_samp[:,0] *= H
p_samp[:,1] *= W

vals = Im_int(p_samp)

V_free = p_samp[vals<THRESH]
###################
# PRM
###################
#p_start = np.array([26,1])

#V_free = np.concatenate((p_start[np.newaxis,:],V_free),axis=0)
Nfree = V_free.shape[0]

K = 30

CD = CollisionDetector(Im_int)

MST, edges = DetectTree(V_free, cost, CD,K)
###################
# Plots
###################

plt.figure()
plt.imshow(I, extent=[0, W, H, 0], cmap='gray')
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(I, extent=[0, W, H, 0], cmap='gray')
plt.colorbar()
for i in range(Nsamp):
    if vals[i] < THRESH:
        color = 'b'

        plt.scatter(p_samp[i,1], p_samp[i,0], color=color)

for t in MST:
    id1 = t[0]

    id2 = t[1]

    p1 = V_free[id1]
    p2 = V_free[id2]

    plt.plot([p1[1],p2[1]], [p1[0],p2[0]], color='r')
plt.show()
