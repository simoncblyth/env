#!/usr/bin/env python
"""

::

   ipython demo_collada_to_chroma.py demo.dae -i 

::

    G4MaterialPropertiesTable* G4Material::GetMaterialPropertiesTable()


* :google:`GetMaterialPropertiesTable wavelength`


"""
import os, sys, logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)   # chroma has weird logging, forcing this placement 

import collada
from chroma.geometry import Solid, Geometry, Mesh 

if __name__ == '__main__':
   if len(sys.argv) > 1:
       path = sys.argv[1]
   else:     
       path = '$LOCAL_BASE/env/geant4/geometry/xdae/g4_01.dae'
   pass    
   path = os.path.expandvars(path)
   log.info("parsing %s " % path )    
   dae = collada.Collada(path)
   log.info("dae %s " % dae ) 

   for g in dae.geometries:
       for p in g.primitives:
           log.info("p %s " % p )












