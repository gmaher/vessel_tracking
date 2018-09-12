import numpy as np

def point_to_index(i,j,H,W):
    return j+i*W

def index_to_point(p,H,W):
    j = p%W

    i = int((p-j)/W)

    return np.array([i,j])

class ImageEnv2D:
    def __init__(self, image, start):
        self.image = image
        self.start = start
        self.s     = self.start.copy()
        self.done  = False

        self.H, self.W = self.image.shape

    def step(action):
        if (action == 0):
            self.s[0] -= 1

        elif (action == 1):
            self.s[1]+=1

        elif (action == 2):
            self.s[2]+=1

        elif (action == 3):
            self.s[3]-=1

        if (self.s[0] == 0 or self.s[0] == H\
            or self.s[1] == 0 or self.s[1] == W):

            self.done = True

        p = point_to_index(self.s[0], self.s[1], self.H, self.W)

        return p, self.done
        
    def reset():
        self.s    = self.start.copy()
        self.done = False;
        return self.s
