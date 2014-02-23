
import numpy as np
import numpy as np
#np.set_printoptions(threshold=50)   # eliding mid elements of large arrays 

standard_wavelengths = np.arange(60, 810, 20).astype(np.float32)
hc_over_GeV = 1.2398424468024265e-06 # h_Planck * c_light / GeV / nanometer #  (approx, hc = 1240 eV.nm )  
hc_over_MeV = hc_over_GeV*1000.      
hc_over_eV  = hc_over_GeV*1.e9


def as_optical_property_vector( s, xunit='MeV', yunit=None ):
    """
    Units of the input string first column as assumed to be MeV, 
    (G4MaterialPropertyVector raw numbers photon energies are in units of MeV)
    these are converted to nm and the order is reversed in the returned
    numpy array.

    :param s: string with space delimited floats representing a G4MaterialPropertyVector 
    :return: numpy array with nm
    """
    # from chroma/demo/optics.py 
    a = np.fromstring(s, dtype=float, sep=' ')
    assert len(a) % 2 == 0
    b = a.reshape((-1,2))[::-1]   ## reverse energy, for ascending wavelength nm

    if yunit is None or yunit in ('','mm'):
        val = b[:,1]
    elif yunit == 'cm':
        val = b[:,1]*10.
    else:   
        assert 0, "unexpected yunit %s " % yunit

    energy = b[:,0]

    if xunit=='MeV':
        e_nm  = hc_over_MeV/energy
    elif xunit=='eV':
        e_nm  = hc_over_eV/energy
    else:
        assert 0, "unexpected xunit %s " % xunit

    vv = np.column_stack([e_nm,val])
    return vv


# from chroma/gpu/geometry.py
def interp_material_property(wavelengths, property):
    # note that it is essential that the material properties be
    # interpolated linearly. this fact is used in the propagation
    # code to guarantee that probabilities still sum to one.
    return np.interp(wavelengths, property[:,0], property[:,1]).astype(np.float32)

def interpolate_check( opv , wavelengths=None):
    """
    Many NuWa properties are defined for wavelengths 200-800nm 
    but the chroma standard wavelengths start from 60nm, 
    this leads to flat properties from 60-200nm 
    using the 200nm value.  

    That might cause problems ? 
    Perhaps should adjust standard wavelengths to 200:800 nm ?
    to match the range where NuWa properties are defined.

    ::

        g4pb:materials blyth$ ./material_properties.py g4_00.dae.6
        ...
        __dd__Materials__OwsWater0xabb2118.ABSLENGTH : (314) 
        [[ 196.00020975  273.208     ]
         [ 197.00020287  369.628     ]
         [ 198.00001706  491.566     ]
         ..., 
         [ 788.00206356  482.415     ]
         [ 790.00041213  486.647     ]
         [ 800.00157879  486.681     ]]
        [[  60.          273.20800781]
         [  80.          273.20800781]
         [ 100.          273.20800781]
         ..., 
         [ 760.          371.97369385]
         [ 780.          425.70596313]
         [ 800.          486.68099976]]


    """ 

    if wavelengths is None:
        wavelengths = standard_wavelengths

    iv = interp_material_property( wavelengths, opv )
    rv = np.column_stack([wavelengths,iv])
    return rv
    
