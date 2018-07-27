import sys
import os
sys.path.append(os.path.abspath('..'))

import numpy as np

from vessel_tracking import geometry

o = np.array([1,2,3])
d = np.array([1,0,0])
step_size = 2
n = 10
bidirectional = True

ray = geometry.ray3(o,d,step_size,n, bidirectional)
