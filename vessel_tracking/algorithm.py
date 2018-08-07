import numpy as np
from vessel_tracking import geometry, signal, sample, ransac

class VesselTracker(object):
    def set_image(self, image):
        self.I_int = image

    def get_path(self):
        pass

    def get_next_point(self):
        pass

class RansacVesselTracker(VesselTracker):
    def set_params(self, Nr, Np, Nd, Nc, p_in, max_dev, inlier_factor, step_size,
        height_step=0.75, max_iter=20):
        self.Nr   = Nr
        self.Np   = Np
        self.Nd   = Nd
        self.Nc   = Nc
        self.p_in          = p_in
        self.max_dev       = max_dev
        self.inlier_factor = inlier_factor
        self.step_size     = step_size
        self.height_step   = height_step
        self.max_iter      = max_iter

    def get_path(self, d, x, r, h):
        cylinders = []
        t = (d,x,r,h,1,[])
        cylinders.append(t)
        count = 0
        while count < self.max_iter:
            t = self.get_next_point(d,x,r,h)

            if t==None: break

            cylinders.append(t)
            print(t[1])

            d=t[0]
            x=t[1]
            r=t[2]
            h=t[3]

            count+=1

        return cylinders

    def get_next_point(self, d0, x0, r0, h0):

        x = x0 + self.height_step*h0*d0*1.0/2
        d = d0
        r = r0
        h = h0

        d,x,r,p,in_,out_ = self.get_ransac_cylinder(d,x,r)

        d,r = geometry.powell_cylinder_3(in_,x,d,r,50)

        coeff = geometry.project_points_on_line(in_, x, d)

        coeff = sorted(coeff)

        UP = int(0.75*len(coeff))
        LOW = int(0.25*len(coeff))
        h = coeff[UP]-coeff[LOW]

        x = x+np.median(coeff)*d

        moved_distance = np.sqrt(np.sum((x-x0)**2))

        if p > self.p_in/2:
            if moved_distance > 0.1*r0:
                return d,x,r,h,p,in_

        print("insufficient progress made, stopping path")
        return None

    def get_ransac_cylinder(self, d0, x0, r0):

        step  = np.sqrt(self.step_size*r0)/self.Np

        surface_points = ransac.sample_surface_points(
            self.I_int, x0, step, self.Nr, self.Np
            )

        #get cylinder
        candidate_axes = sample.sphere_sample_vec(d0, self.Nd)

        curr_best_p = 0
        curr_best_d = 0
        curr_best_c = 0
        curr_best_r = 0
        curr_in = 0
        curr_out = 0

        for i in range(self.Nd):
            dtest = candidate_axes[i]

            best_center, best_r, best_in_rate, inliers, outliers = \
                ransac.ransac_cylinder(surface_points, x0, dtest, r0, self.p_in,
                    self.max_dev, self.inlier_factor, self.Nc)

            if best_in_rate > curr_best_p:
                curr_best_p = best_in_rate
                curr_best_d = dtest
                curr_best_c = best_center
                curr_best_r = best_r
                curr_in     = inliers
                curr_out    = outliers

            if best_in_rate > self.p_in and ( (best_r <= (1+self.max_dev)*r0)\
                and (best_r > (1-self.max_dev)*r0) ):

                print("acceptable cylinder found {}, {}, {}, {}"\
                    .format(curr_best_d, curr_best_c, curr_best_r, curr_best_p))

                break

        return curr_best_d, curr_best_c, curr_best_r, curr_best_p, curr_in, curr_out
