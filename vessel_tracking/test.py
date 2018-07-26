import geometry
import numpy as np

v1 = np.array([1,0,0])
v2 = np.array([1,1,1])

print("orth: {}, {}".format(v1,geometry.orth(v1)))
print("orth: {}, {}".format(v2,geometry.orth(v2)))

print("perpendicular_plane: {}, {}".format(v1,geometry.perpendicular_plane(v1)))
print("perpendicular_plane: {}, {}".format(v2,geometry.perpendicular_plane(v2)))

#test projection

p  = np.array([[1,1,1]])
o  = np.array([0,0,0])
q1 = np.array([1,0,0])
q2 = np.array([0,1,0])

coeffs = geometry.project_points_into_plane(p,o,q1,q2)

print("project_points_into_plane {} {} {} {} {}".format(p,o,q1,q2,coeffs))

p  = np.array([[1,1,1]])
o  = np.array([2,2,2])
q1 = np.array([1,0,0])
q2 = np.array([0,1,0])

coeffs = geometry.project_points_into_plane(p,o,q1,q2)

print("project_points_into_plane {} {} {} {} {}".format(p,o,q1,q2,coeffs))

p  = np.array([[1,1,1], [-1,-1,-1]])
o  = np.array([2,2,2])
q1 = np.array([1,0,0])
q2 = np.array([0,1,0])

coeffs = geometry.project_points_into_plane(p,o,q1,q2)

print("project_points_into_plane {} {} {} {} {}".format(p,o,q1,q2,coeffs))


X = np.array(
[
[1,0,],
[0,1,],
[-1,0,]
]
)

print("fit_circle_3 {} {}".format(X, geometry.fit_circle_3(X)))

X = np.array(
[
[2,0],
[0,2],
[-2,0]
]
)

print("fit_circle_3 {} {}".format(X, geometry.fit_circle_3(X)))

X = np.array(
[
[4,0],
[2,2],
[0,0]
]
)

print("fit_circle_3 {} {}".format(X, geometry.fit_circle_3(X)))

X = np.array(
[
[1,0,0.5],
[0,1,-1],
[np.sqrt(0.5), np.sqrt(0.5),1]
]
)

o = np.array([0,0,0])
d = np.array([0,0,1])

print("fit_cylinder_3 {} {} {} {}".format(X, o, d, geometry.fit_cylinder_3(X,o,d)))

X = np.array(
[
[1,0,0.5],
[0,1,-1],
[np.sqrt(0.5), np.sqrt(0.5),1]
]
)

o = np.array([0,0,2])
d = np.array([0,0,1])

print("fit_cylinder_3 {} {} {} {}".format(X, o, d, geometry.fit_cylinder_3(X,o,d)))



o = np.array([1,1,1])
d = np.array([0,1,1])

q1 = np.array([1,0,0])
q2 = np.array([0,-np.sqrt(0.5), np.sqrt(0.5)])

X = np.zeros((3,3))
X[0,:] = o + 0.5*d + 3*q1 + 4*q2
X[1,:] = o - 0.5*d + 5*q1
X[2,:] = o + 2*d - 4*q1 - 3*q2

print("fit_cylinder_3 {} {} {} {}".format(X, o, d, geometry.fit_cylinder_3(X,o,d)))

print("distance_in_plane {} {} {} {}".format(X,o,d, geometry.distance_in_plane(X,o,d)))
