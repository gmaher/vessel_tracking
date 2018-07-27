import geometry
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

noise_factor  = 1

points = points + np.random.randn(800,3)*noise_factor

#ransac
p_in          = 0.7
N_iter        = 100
best_in_rate  = 0
max_deviation = 0.5
inlier_dist   = 0.1*r
n_points      = points.shape[0]
index         = np.arange(0,n_points)

best_center = 0
best_r      = 0

q1,q2 = geometry.perpendicular_plane(d)

for i in range(N_iter):
    inds = np.random.choice(n_points,size=3,replace=False)
    oos_inds = [i for i in index if not any([i==h for h in inds])]

    X = points[inds]
    X_oos = points[oos_inds]

    try:
        c_new, r_new = geometry.fit_cylinder_3(X, o, d)
    except:
        print("failed to fit cylinder {}".format(X))

    distances = geometry.distance_in_plane(X_oos,c_new, d)

    in_out = np.abs(distances-r) <= inlier_dist

    in_rate = np.sum(in_out)*1.0/(n_points-3)

    if in_rate > best_in_rate and np.abs(r_new-r) < max_deviation*r:
        best_in_rate = in_rate

        best_center  = c_new
        best_r       = r_new

        inliers = X_oos[in_out]
        outliers = X_oos[~in_out]

print(best_center,r)

Ps = geometry.cylinder_surface(-1,1,20, best_center, d, best_r)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Ps[:,:,0], Ps[:,:,1], Ps[:,:,2])

ax.scatter(inliers[:,0],inliers[:,1],inliers[:,2],color='g')
ax.scatter(outliers[:,0],outliers[:,1],outliers[:,2],color='r')
plt.show()
plt.close()
