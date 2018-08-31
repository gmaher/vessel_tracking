import numpy as np
import os
import sys
import SimpleITK as sitk
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('../../..'))

from vessel_tracking import util

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

###################
# Sim
###################
Nsim   = 1000
Nsteps = 100
eps = 0.75
gamma = 1.0

x_start = 1
y_start = N//2

V = np.zeros((N,N))

Visits = np.zeros((N,N))+1e-7
Visits[0,:] = 1e7
Visits[N-1,:] = 1e7
Visits[:,0] = 1e7
Visits[:,N-1] = 1e7

#Cp = 1.0/np.sqrt(2)
Cp = 100
lr = 0.1

for i in range(Nsim):
    x = x_start
    y = y_start

    print ("Sim {}".format(i))

    visited = {}

    for j in range(Nsteps):

        print("step {}, y={} x={}".format(j,y,x))

        beta = 1
        if (y,x) in visited:
            beta = 0.1

        visited[(y,x)] = 1

        #do ucb stuff
        Visits[y,x] += 1
        cv = np.log(Visits[y,x]+1e-7)

        vis = [ Visits[y-1,x],
            Visits[y,x+1],
            Visits[y+1,x],
            Visits[y,x-1] ]

        values = [ V[y-1,x] + Cp*np.sqrt(cv/Visits[y-1,x]),
            V[y,x+1]+ Cp*np.sqrt(cv/Visits[y,x+1]),
            V[y+1,x]+ Cp*np.sqrt(cv/Visits[y+1,x]),
            V[y,x-1]+ Cp*np.sqrt(cv/Visits[y,x-1]) ]

        a = np.argmax(values)
        # print(values, a)
        # print("visits={}".format(vis))
        # print(Visits[y,x])
        mov = action_to_move(a)

        xx = x+mov[0]
        yy = y+mov[1]

        if xx < 1:   xx = 1
        if xx > N-2: xx = N-2
        if yy < 1:   yy = 1
        if yy > N-2: yy = N-2

        r = beta*vessel_func(yy,xx,I)

        V[y,x] = (1-lr)*V[y,x]+ lr*(r + gamma*V[yy,xx])

        x = xx
        y = yy

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

plt.figure()
plt.imshow(V,cmap='gray')
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(Visits[1:-1,1:-1],cmap='gray')
plt.colorbar()
plt.show()

################################
# Sample Paths
################################
Nsim   = 1000
Nsteps = 200
paths = []

x_start = 1
y_start = N//2+1

for i in range(Nsim):
    print(i)
    x = x_start
    y = y_start
    path = []
    path.append((x,y))

    for s in range(1,Nsteps):
        values = np.array([ V[y-1,x], V[y,x+1], V[y+1,x], V[y,x-1] ])
        values = values/(np.sum(values))

        a = np.random.choice(4, p=values)

        mov = action_to_move(a)

        x = x+mov[0]
        y = y+mov[1]

        path.append((x,y))
        if x == 0 or x == N-1 or y == 0 or y == N-1:
            break

    paths.append(path)


plt.figure()
plt.imshow(I, extent=[0,N,N,0], cmap='gray')
plt.colorbar()

for p in paths:
    x = [z[0] for z in p]
    y = [z[1] for z in p]

    plt.plot(x,y)

plt.show()
