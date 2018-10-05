import numpy as np
import os
import sys
import SimpleITK as sitk
import matplotlib.pyplot as plt
from queue import Queue

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
            if float(self.I_int(p)) <= 0.1:
                return True
        return False

def vessel_func(i,j,I):
    return np.mean(I[i-1:i+1,j-1:j+1])

def cost(p,pp):
    return np.linalg.norm(p-pp)

N = 50
NOISE_FACTOR = 0.35

Noise = np.random.randn(N,N)*NOISE_FACTOR

counts = np.zeros((N,N))

I = np.zeros((N,N))

x = np.linspace(-1,1,N)
X,Y = np.meshgrid(x,x)

#create image
I[N//2, 0:N//3]   = 1.0
I[N//2+1, 0:N//3] = 1.0
I[N//2+2, 0:N//3] = 1.0
I[N//2+3, 0:N//3] = 1.0


I[N//10:N//2, N//5-1] = 1.0
I[N//10:N//2, N//5-2] = 1.0

I[N//3, N//5-1:N//2] = 1.0

I[N//3-2, N//5-1:int(N*3/4)] = 1.0

I[N//3:N//2, N//2-5] = 1.0

I[N//2:int(N*0.9), N//5-5] = 1.0
I[N//2:int(N*0.9), N//5-6] = 1.0

I[int(N*0.75), N//5-5:int(0.9*N)] = 1.0

I_vec = np.zeros((N*N))
points = np.zeros((N*N,2))
for i in range(N):
    for j in range(N):
        points[j+i*N,0] = i
        points[j+i*N,1] = j

        I_vec[j+i*N] = I[i,j]


from scipy.interpolate import LinearNDInterpolator

Im_int = LinearNDInterpolator(points+0.5, np.ravel(I))

Nsamp = 2000
p_samp = np.random.rand(Nsamp,2)*N
vals = Im_int(p_samp)

V_free = p_samp[vals>0]
###################
# PRM
###################
p_start = np.array([26,1])

V_free = np.concatenate((p_start[np.newaxis,:],V_free),axis=0)
Nfree = V_free.shape[0]

K = 30

CD = CollisionDetector(Im_int)

MST, edges = DetectTree(V_free, cost, CD,K)
###################
# Plots
###################

plt.figure()
plt.imshow(I, extent=[0, N, N, 0], cmap='gray')
plt.colorbar()

for i in range(Nsamp):
    if vals[i] > 0:
        color = 'b'

        plt.scatter(p_samp[i,1], p_samp[i,0], color=color)

for t in MST:
    id1 = t[0]

    id2 = t[1]

    p1 = V_free[id1]
    p2 = V_free[id2]

    plt.plot([p1[1],p2[1]], [p1[0],p2[0]], color='r')
plt.show()
