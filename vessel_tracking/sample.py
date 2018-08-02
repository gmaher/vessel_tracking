import numpy as np

def sphere_sample(N=1, n=3):
    x = np.random.randn(N,n)

    lengths = np.sqrt(np.sum(x**2,axis=1))[:,np.newaxis]

    x = x/lengths

    return x

def sphere_sample_vec(d, N=1):
    FACTOR = 3
    n = d.shape[0]

    z = sphere_sample(int(FACTOR*N), n)

    length = np.sqrt(np.sum(d**2))

    v = d*1.0/length

    dots = np.sum(z*v, axis=1)

    angles = np.arccos(dots)

    zipped = [(z[i], angles[i]) for i in range(int(FACTOR*N))]

    zipped = sorted(zipped, key=lambda x: x[1])

    vecs = np.array([t[0] for t in zipped])*length

    return vecs
