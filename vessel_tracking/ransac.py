from vessel_tracking import geometry, sample, signal
import numpy as np

def sample_surface_points(I_int, c, step_size, Nr=1, Np=50):
    directions = sample.sphere_sample(Nr)
    surface_points = []

    for i in range(Nr):
        d = directions[i]
        ray = geometry.ray(c,d,step_size,Np, bidirectional=False)

        intensities = signal.smooth_n(I_int(ray), N=2)

        grad        = signal.central_difference(intensities)

        ind = np.argmin(grad)

        surface_points.append(ray[ind])


    surface_points = np.array(surface_points)

    return surface_points

def ransac_cylinder(points, o, d, r, p_in=0.7, max_deviation=0.5,
    inlier_factor=0.1, N_iter=100):
    """
    fits a cylinder using  ransac
    args:
        points - points nx3 array
        o - origin (3) array
        d - cylinder axis vector (3) array
        r - initial radius estimate
        p_in - minimum inlier rate to accept
        Niter - max number of tries

    """
    inlier_dist   = inlier_factor*r
    n_points      = points.shape[0]
    index         = np.arange(0,n_points)

    best_center  = 0
    best_r       = 0
    best_in_rate = 0

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

        in_out = np.abs(distances-r_new) <= inlier_dist

        in_rate = np.sum(in_out)*1.0/(n_points-3)

        if in_rate > best_in_rate and np.abs(r_new-r) < max_deviation*r:
            best_in_rate = in_rate

            best_center  = c_new
            best_r       = r_new

            inliers = X_oos[in_out]
            outliers = X_oos[~in_out]

    return best_center, best_r, best_in_rate, inliers, outliers
