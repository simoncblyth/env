#!/usr/bin/env python
"""
Compare TransformCache and IDMAP Transforms
==============================================

TransformCache
    Holds a map of transforms::

        map<size_t, G4AffineTransform> id2transform  
 
   And optionally can construct 
   binary format persisted from G4AffineTransform data (read from GDML loaded geometry)
   used to access the transforms to calc local coordinates of hits 
   
   The persisting is useful for low dependendency testing, for actual running 
   the in memory map is sufficient::

IDMAP
   text formatted from G4AffineTransform data from original in memory geometry 



Issues
-------

TODO
~~~~~

Drop or improve formatting of transforms dumped to the idmap ? 
they were an afterthought. 


Can IDMAP be eliminated ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The IDMAP is a geometry level thing that needs to be 
communicated to the GPU via the idmaplink mechanicsm 
of g4daenode.py which is used in formation of chroma geometry.

Wherease the TransformCache is a 
Do that within the TransformCache step ? 



IDMAP Observations
~~~~~~~~~~~~~~~~~~

The principal role of the IDMAP is the integer mapping between volume index 
and PMTID for SD.  Current IDMAP has PMTID labels for many more
volumes than just the Cathode SD.  All volumes regarded to be within 
the DE are labelled.

Discrepancy for all HeadOn
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    {'/dd/Geometry/PMT/lvHeadonPmtGlass#pvHeadonPmtVacuum': 12,
     '/dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum': 672} 684

All 12 HeadonPmtVacuum show discrepancy between the transforms 
via the idmap and those persisted in the transform cache. 
Looks like a numerical formatting in the text based idmap 
is loosing precision for large x and y values versus
the binary based transform cache which is not susceptible 
to formatting truncations.


::

    (chroma_env)delta:collada blyth$ ipython check_transform_cache.py 

    /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum
    /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum
    /dd/Geometry/PMT/lvHeadonPmtGlass#pvHeadonPmtVacuum
    0
    (4353, 16842753, '1010001', '(668899,-439222,-4909)', '(0.567844,0.823136,-6.95385e-17)(0.823136,-0.567844,-1.00802e-16)(-1.22461e-16,6.16298e-33,-1)', '/dd/Geometry/PMT/lvHeadonPmtGlass#pvHeadonPmtVacuum')
    ttc [[      0.568       0.823      -0.     668898.507]
     [      0.823      -0.568      -0.    -439222.475]
     [     -0.         -0.         -1.      -4961.5  ]
     [      0.          0.          0.          1.   ]]
    itr [[      0.568       0.823      -0.     668899.   ]
     [      0.823      -0.568      -0.    -439222.   ]
     [     -0.          0.         -1.      -4909.   ]
     [      0.          0.          0.          1.   ]]
    /dd/Geometry/PMT/lvHeadonPmtGlass#pvHeadonPmtVacuum
    1
    (4360, 16842754, '1010002', '(-668899,-439222,9276)', '(-0.567844,0.823136,0)(-0.823136,-0.567844,0)(0,0,1)', '/dd/Geometry/PMT/lvHeadonPmtGlass#pvHeadonPmtVacuum')
    ttc [[     -0.568       0.823       0.    -668898.507]
     [     -0.823      -0.568       0.    -439222.475]
     [      0.          0.          1.       9223.5  ]
     [      0.          0.          0.          1.   ]]
    itr [[     -0.568       0.823       0.    -668899.   ]
     [     -0.823      -0.568       0.    -439222.   ]
     [      0.          0.          1.       9276.   ]
     [      0.          0.          0.          1.   ]]
    /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum




"""
import numpy as np

from idmap import IDMap
from transform_cache import TransformCache 

if __name__ == '__main__':
    idmap = IDMap()
    tc = TransformCache()
    np.set_printoptions(suppress=True, precision=3 )

    count = 0  
    pvn = {} 
    for k in tc:
        name = idmap.a[k]['pvname']
        if not name in pvn:
            pvn[name] = 0
        pvn[name] += 1 

        assert 3000 < k < 13000, "expecting volume index : not PmtId"
        ttc = tc[k]
        itr = idmap.transform[k]
        if not np.allclose( ttc, itr ):
            print count
            print idmap.a[k]
            print "ttc", ttc
            print "itr", itr
            count += 1 
        pass
    pass
    print pvn, len(tc)


