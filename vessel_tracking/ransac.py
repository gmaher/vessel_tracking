def ransac_cylinder(P, o, d, p_in=0.7, Niter=100):
    """
    fits a cylinder using  ransac
    args:
        P - points nx3 array
        o - origin (3) array
        d - cylinder axis vector (3) array
        p_in - minimum inlier rate to accept
        Niter - max number of tries

    returns:
        o_new - new cylinder center
        r_new - cylinder radius
        p_in_new - inlier rate
    """
    pass
