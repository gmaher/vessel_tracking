import numpy as np
import os
import sys
import SimpleITK as sitk
import matplotlib.pyplot as plt

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
for i in range(N):
    for j in range(N):
        I_vec[j+i*N] = I[i,j]

###################
# Sim
###################
Nsim   = 10000
Nsteps = 300
eps = 0.9
gamma = 0.99
lr = 0.1

i_start = N//2+1
j_start = 1
s_start = np.array([i_start,j_start])

index_start = rl.point_to_index(s_start[0],s_start[1],N,N)

env = rl.ImageEnv2D(I,s_start)

Q   = np.zeros((N*N,4))

for n in range(Nsim):

    s = env.reset()

    print ("Sim {}".format(n))

    for t in range(Nsteps):

        rand = np.random.rand()
        if rand < eps:
            a = np.random.randint(4)
        else:
            a = np.argmax(Q[s])

        ss,done = env.step(a)

        r = I_vec[ss]

        Q[s,a] = (1-lr)*Q[s,a] + lr*( r + gamma*np.amax( Q[ss] ) )

        print("step {}, s={}, a={}, r={}, ss={}, Q={}".format(t,s,a,r,ss, Q[s,a]))
        s = ss

        if(done):
            break

Qp = np.amax(Q,axis=1)
Qp = np.reshape(Qp,(N,N))
###################
# Plots
###################
plt.figure()
plt.imshow(I,cmap='gray')
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(Qp,cmap='gray')
plt.colorbar()
plt.show()
#
# plt.figure()
# plt.imshow(V,cmap='gray')
# plt.colorbar()
# plt.show()
#
# plt.figure()
# plt.imshow(Visits[1:-1,1:-1],cmap='gray')
# plt.colorbar()
# plt.show()
