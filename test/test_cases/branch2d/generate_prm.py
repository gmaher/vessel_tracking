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

Nsamp = 1000
p_samp = np.random.randint(N,size=(Nsamp,2))
vals = Im_int(p_samp)

V_free = p_samp[vals>0]
###################
# PRM
###################
edges = np.zeros((Nsamp,Nsamp))
q = Queue()
p_start = np.array([26,1])
K = 3
q.put(p_start)
V = V_free.copy()
while not q.empty():
    p = q.get()

    dists = np.sum((V_free-p)**2,axis=1)

    d_sorted = sorted(dists)
###################
# Plots
###################
plt.figure()
plt.imshow(I,cmap='gray')
plt.colorbar()
plt.show()


plt.figure()
plt.imshow(I,cmap='gray')
plt.colorbar()

for i in range(Nsamp):
    if vals[i] > 0:
        color = 'r'

        plt.scatter(p_samp[i,1], p_samp[i,0], color=color)
plt.show()
