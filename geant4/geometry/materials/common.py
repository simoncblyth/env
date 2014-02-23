
import numpy as np
import numpy as np
np.set_printoptions(threshold=5)   # eliding mid elements of large arrays 


def as_optical_property_vector( s ):
    """
    Units of the input string first column as assumed to be MeV, 
    (G4MaterialPropertyVector raw numbers photon energies are in units of MeV)
    these are converted to nm and the order is reversed in the returned
    numpy array.

    :param s: string with space delimited floats representing a G4MaterialPropertyVector 
    :return: numpy array with nm
    """
    # from chroma/demo/optics.py 
    hc_over_GeV = 1.2398424468024265e-06 # h_Planck * c_light / GeV / nanometer #  (approx, hc = 1240 eV.nm )  
    hc_over_MeV = hc_over_GeV*1000.      

    a = np.fromstring(s, dtype=float, sep=' ')
    assert len(a) % 2 == 0
    b = a.reshape((-1,2))[::-1]   ## reverse energy, for ascending wavelength nm

    v = b[:,1]
    e_mev = b[:,0]
    e_nm  = hc_over_MeV/e_mev
    vv = np.column_stack([e_nm,v])
    return vv



