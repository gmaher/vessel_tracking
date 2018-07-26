import geometry
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

start = -1
end_  = 1
N     = 20

d = np.array([1,1,1])
d = d/np.sqrt(np.sum(d**2))

q1,q2 = geometry.perpendicular_plane(d)
r = 2
o = np.array([2,2,2])

x = np.linspace(-1,1,N)
z = np.linspace(start,end_,N)

xm,zm = np.meshgrid(x,z)
ym = np.sqrt(1-xm**2)

X = np.concatenate((xm,np.fliplr(xm)),axis=1)
Y = np.concatenate((ym,-ym),axis=1)
Z = np.concatenate((zm,zm),axis=1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X,Y,Z)
plt.show()
plt.close()

Nm = X.shape[0]

xv = np.ravel(X)
yv = np.ravel(Y)
zv = np.ravel(Z)

n = len(xv)

V = np.zeros((n,3))
V[:,0] = xv
V[:,1] = yv
V[:,2] = zv

Q = np.zeros((3,3))
Q[0,:] = q1
Q[1,:] = q2
Q[2,:] = d

P = V.dot(Q) + o

Ps = P.reshape((Nm,2*Nm,3))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Ps[:,:,0], Ps[:,:,1], Ps[:,:,2])
plt.show()
plt.close()
