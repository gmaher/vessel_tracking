import numpy as np
import os
import sys
import SimpleITK as sitk
import matplotlib.pyplot as plt
from queue import Queue

sys.path.append(os.path.abspath('../../..'))

from vessel_tracking import util, rl

def vessel_func(i,j,I):
    return np.mean(I[i-1:i+1,j-1:j+1])

def action_to_move(a):
    if a == 0:
        return (0,-1)
    if a == 1:
        return (1,0)
    if a == 2:
        return (0,1)
    if a == 3:
        return (-1,0)

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

Im_int = LinearNDInterpolator(points, np.ravel(I))

Nsamp = 2000
p_samp = np.random.randint(N,size=(Nsamp,2))
vals = Im_int(p_samp)

V_free = p_samp[vals>0]
###################
# PRM
###################
p_start = np.array([26,1])

V_free = np.concatenate((p_start[np.newaxis,:],V_free),axis=0)
Nfree = V_free.shape[0]

q = Queue()
K = 3
Kfar = 10
q.put(0)
DIST_CUTOFF=50

connected=False

in_tree = np.zeros((V_free.shape[0]))
in_tree[0] = 1

edges = [ 0 ]*Nfree
for i in range(Nfree):
    edges[i] = []
#TODO: collision free check needed
while not q.empty():
    p = q.get()
    connected = False

    dists = np.sum((V_free-V_free[p])**2,axis=1)

    dists[p] = 1e10

    idx   = np.argpartition(dists,Kfar)[:Kfar]

    print("{} : {}".format(p,idx))
    for i in range(K):
        if in_tree[idx[i]] == 0:
            edges[p].append(idx[i])
            q.put(idx[i])
            in_tree[idx[i]] = 1
            connected = True

    if not connected:
        for i in range(Kfar):
            if in_tree[idx[i]] == 0:
                if (dists[idx[i]] < DIST_CUTOFF):
                    edges[p].append(idx[i])
                    q.put(idx[i])
                    in_tree[idx[i]] = 1
                    break

###################
# Plots
###################
plt.figure()
plt.imshow(I,cmap='gray')
plt.colorbar()
plt.show()


plt.figure()
plt.imshow(I, extent=[0, N, N, 0], cmap='gray')
plt.colorbar()

for i in range(Nsamp):
    if vals[i] > 0:
        color = 'b'

        plt.scatter(p_samp[i,1], p_samp[i,0], color=color)

for i in range(Nfree):
    p1 = V_free[i]
    for e in edges[i]:
        p2 = V_free[e]

        plt.plot([p1[1],p2[1]], [p1[0],p2[0]], color='r')
plt.show()
