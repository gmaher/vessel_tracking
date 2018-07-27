import sys
import os
sys.path.append(os.path.abspath('..'))

from vessel_tracking import ransac, geometry

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

start = -1
end_  = 1
N     = 20

d = np.array([1,1,-1])
d = d/np.sqrt(np.sum(d**2))

q1,q2 = geometry.perpendicular_plane(d)
r = 2
o = np.array([2,2,-2])

Ps = geometry.cylinder_surface(-1,1,20, o, d, r)

points = Ps.reshape((800,3))

noise_factor  = 0.25

points = points + np.random.randn(800,3)*noise_factor

#ransac
p_in          = 0.7
max_deviation = 0.5
inlier_factor = 0.1
N_iter        = 100

best_center, best_r, best_in_rate, inliers, outliers = ransac.ransac_cylinder(
    points, o, d, r, p_in=p_in, max_deviation=max_deviation,
    inlier_factor=inlier_factor, N_iter=N_iter
)

print(best_center,best_r)

Ps = geometry.cylinder_surface(-1,1,20, best_center, d, best_r)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Ps[:,:,0], Ps[:,:,1], Ps[:,:,2])

ax.scatter(inliers[:,0],inliers[:,1],inliers[:,2],color='g')
ax.scatter(outliers[:,0],outliers[:,1],outliers[:,2],color='r')
plt.show()
plt.close()
