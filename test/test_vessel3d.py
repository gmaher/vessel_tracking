import sys
import os
sys.path.append(os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import LinearNDInterpolator

from vessel_tracking import geometry, signal

def vessel(x,r):
    if np.sqrt(x[0]**2+x[1]**2)<=r:
        return 1
    else:
        return 0

N = 100
HEIGHT  = 5
WIDTH   = 5
DEPTH   = 5
SPACING = HEIGHT*1.0/N

RADIUS  = 1
