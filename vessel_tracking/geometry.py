import numpy as np

def orth(v):
    """
    return vector orthogonal to v
    """

    if v[0] == 0:
        q = np.array([1,0,0])

    else:
        q = np.zeros((3))

        q[0]= -(v[1]+v[2])/v[0]
        q[1]=1
        q[2]=1

        q = q/np.sqrt(np.sum(q**2))

    return q

def perpendicular_plane(v):
    """
    return two vectors making perpendicular plane to v
    """
    q1 = orth(v)

    q2 = np.cross(v,q1)
    q2 = q2/np.sqrt(np.sum(q2**2))

    return q1,q2

def project_points_into_plane(P,o,q1,q2):
    """
    projects points P into plane spanned by origin o, and vecs q1,q2
    args:
        P - nx3 array
    """
    if not (len(P.shape) == 2) or not P.shape[1] == 3:
        raise RuntimeError("P must be nx3, currently {}".format(P.shape))

    Z = P-o

    coeff_1 = Z.dot(q1)
    coeff_2 = Z.dot(q2)

    X = np.zeros((P.shape[0],2))
    X[:,0] = coeff_1
    X[:,1] = coeff_2

    return X

def distance_in_plane(P,o,d):
    """
    calculates perpendiular distance of points in P to line o+alpha*d
    args:
        P - nx3 array
    """
    if not (len(P.shape) == 2) or not P.shape[1] == 3:
        raise RuntimeError("P must be nx3, currently {}".format(P.shape))

    q1,q2 = perpendicular_plane(d)

    X_plane = project_points_into_plane(P,o,q1,q2)

    return np.sqrt(np.sum(X_plane**2,axis=1))

def fit_circle_3(P):
    """
    fits a circle to 3 points in a plane (Beder 2006)
    args:
        P - nx2 array
    """
    if not (len(P.shape) == 2) or not P.shape[1] == 2:
        raise RuntimeError("P must be nx2, currently {}".format(P.shape))

    A = np.zeros((3,3))
    A[:,0] = 2*P[:,0]
    A[:,1] = 2*P[:,1]
    A[:,2] = -1

    b = P[:,0]**2+P[:,1]**2

    x = np.linalg.solve(A,b)

    r = np.sqrt(x[0]**2+x[1]**2-x[2])

    center = x[:2]

    return center, r

def fit_cylinder_3(X, o, d):
    q1,q2   = perpendicular_plane(d)

    X_plane = project_points_into_plane(X,o,q1,q2)

    center, r = fit_circle_3(X_plane)

    center_3d = o + center[0]*q1 + center[1]*q2

    return center_3d,r

def cylinder_surface(start, end, N, o, d, r):
    pass
