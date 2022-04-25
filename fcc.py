import numpy as np 

π = np.pi 

def FCClattice(a, element, la, lb, lc, sphere=False, CoM=np.zeros(3), names=False):
    """
    GIVEN:  a (lattice constant)
            element (atomic number)
            la, lb, lc (number of unit cells in each direction)
            **sphere (option to get ellipsoid cutout)
            **CoM (option to set the Center-of-Mass, default is (0,0,0))
            **names (option to return element names)
    GET:    Z     (array with atomic number)
            R_ix  (all positions in the full crystal)
            m_i   (get mass array)
    """

    Z_mass = np.array([ 1.,   1837.,   7297.,  
                   12650., 16427.,  19705.,  21894.,  25533.,  29164.,  34631., 36785.,  
                   41908., 44305.,  49185.,  51195.,  56462.,  58441.,  64621.,  72820.,  
                   71271., 73057.,  81949.,  87256.,  92861.,  94782., 100145., 101799., 107428., 106990., 115837., 119180., 127097., 132396., 136574., 143955., 145656., 152754.,
                   155798., 159721., 162065., 166291., 169357., 174906., 176820., 184239., 187586., 193991., 196631., 204918., 209300., 216395., 221954., 232600., 231331., 239332., 
                   242270., 250331., 253208., 255415., 256859., 262937., 264318., 274089., 277013., 286649., 289702., 296219., 300649., 304894., 307947., 315441., 318945., 325367., 329848., 335119., 339434., 346768., 350390., 355616., 359048., 365656., 372561., 377702., 380947., 380983., 382806., 404681.,
                   406504., 411972., 413795., 422979., 421152., 433900., 432024., 444784., 442961., 450253., 450253., 457545., 459367., 468482., 470305., 472128., 477596., 486711., 492179., 490357., 492179., 492179., 506763., 512231., 512231., 519523., 521346., 526814., 526814., 534106., 534106., 535929.])

    Atom_Names = np.array(["e ", "H ", "He", 
    "Li", "Be", "B ", "C ", "N ", "O ", "F ", "Ne", 
    "Na", "Mg", "Al", "Si", "P ", "S ", "Cl", "Ar", 
    "K ", "Ca", "Sc", "Ti", "V ", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", 
    "Rb", "Sr", "Y ", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I ", "Xe", 
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W ", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", 
    "Fr", "Ra", "Ac", "Th", "Pa", "U ", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"])


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

    Z = np.tile(element, len(R_ix)).astype(np.int8)
    mass = Z_mass[Z]

    CofM  = mass @ R_ix / np.sum(mass)
    R_ix -= (CofM - CoM)
    if names:
        return Z, R_ix, mass, Atom_Names[Z]
    else:
        return Z, R_ix, mass


