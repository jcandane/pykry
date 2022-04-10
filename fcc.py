import numpy as np 

π = np.pi 

def FCClattice(a, la, lb, lc, sphere=False, element=None):
    """
    GIVEN:  a lattice constant
            la, lb, lc (number of unit cells in each direction)
            **sphere (option to get ellipsoid cutout)
            **element (option to give element type)
    GET:    R_ix, all positions in the full crystal
            **Z, return array with element type
    """

    R_ix_in  = np.array([[0.,0.,0.],[np.sqrt(2)/2,np.sqrt(2)/2,0.],[np.sqrt(2)/2.,0.,np.sqrt(2)/2.],[0., np.sqrt(2)/2.,np.sqrt(2)/2.]])
    R_ix_in *= a/np.sqrt(2)

    x_   = a * np.arange(0, la, 1)
    y_   = a * np.arange(0, lb, 1)
    z_   = a * np.arange(0, lc, 1)
    R_Ix = np.array(np.meshgrid(x_, y_, z_))
    R_Ix = R_Ix.reshape( (3, len(x_)*len(y_)*len(z_)) , order="C").T

    if sphere:
        center = np.array([la*a, lb*a, lc*a])/2
        ll = (np.array([la, lb, lc]) * a/2)**2
        R_Ix = R_Ix[ np.sum( ( R_Ix - center[None,:] )**2 / ll[None,:], axis = 1 ) < 1 ]

    R_ix = (R_ix_in[None, :, :] + R_Ix[:,None,:]).reshape((R_Ix.shape[0]*R_ix_in.shape[0], 3), order="C")

    if element is not None:
        return np.tile(element, len(R_ix)), R_ix
    else:
        return R_ix


