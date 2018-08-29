import numpy as np
import os
import sys
import SimpleITK as sitk
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('../../..'))

from vessel_tracking import util

def vessel_func(i,j,I):
    if i ==0 or i == N-1 or j == 0 or j == N-1:
        return I[i,j]

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

def check_path(p,I, min_val=12):
    #no repeat visits
    d = {}
    for point in p:
        if point in d: return False
        d[point] = 1

    value = 0
    for point in p:
        value += vessel_func(point[1], point[0], I)


    if value < min_val: return False
    print(value)

    return True

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

###################
# Sim
###################
Nsim   = 1000000
Nsteps = 30
paths = []

x_start = 1
y_start = N//2

for i in range(Nsim):
    print(i)
    x = x_start
    y = y_start
    path = []
    path.append((x,y))

    for s in range(1,Nsteps):
        mov_x = np.random.randint(low=-1,high=2)
        mov_y = np.random.randint(low=-1,high=2)

        x = x+mov_x
        y = y+mov_y

        path.append((x,y))
        if x == 0 or x == N-1 or y == 0 or y == N-1:
            break

    paths.append(path)
###################
# Plots
###################
plt.figure()
plt.imshow(I,cmap='gray')
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(I+Noise,cmap='gray')
plt.colorbar()
plt.show()

# plt.figure()
# plt.imshow(I, extent=[0,N,0,N], cmap='gray')
# plt.colorbar()
#
# for p in paths:
#     x = [z[0] for z in p]
#     y = [z[1] for z in p]
#
#     plt.plot(x,y)
#
# plt.show()
#
plt.figure()
plt.imshow(I, extent=[0,N,N,0], cmap='gray')
plt.colorbar()

for p in paths:
    if check_path(p,I):
        x = [z[0] for z in p]
        y = [z[1] for z in p]

        plt.plot(x,y)

plt.show()
