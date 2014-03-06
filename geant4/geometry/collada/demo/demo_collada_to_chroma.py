#!/usr/bin/env python
"""
::

   ipython demo_collada_to_chroma.py demo.dae -i 

"""
import os, sys, logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)   # chroma has weird logging, forcing this placement 

import numpy as np
from env.geant4.geometry.collada.collada_to_chroma import daeload, matshorten
from chroma.geometry import standard_wavelengths

def interp_material_property(wavelengths, property):
    # note that it is essential that the material properties be
    # interpolated linearly. this fact is used in the propagation
    # code to guarantee that probabilities still sum to one.
    return np.interp(wavelengths, property[:,0], property[:,1]).astype(np.float32)


def dump_chroma_materials( geometry, wavelengths=None):
   if not wavelengths:
       wavelengths = standard_wavelengths

   for i in range(len(geometry.unique_materials)):
       material = geometry.unique_materials[i]
       miss = []
       for attn in "refractive_index absorption_length scattering_length reemission_prob reemission_cdf".split():
           assert hasattr(material, attn)
           attr = getattr(material, attn)
           if attr is None:
               miss.append(attn)
           else:
               vals = interp_material_property(wavelengths, attr)
               #print attn, vals
              
       print "%-3s %-25s %s " % ( i, matshorten(material.name), ",".join(miss) )

       #refractive_index = interp_material_property(wavelengths, material.refractive_index)
       #absorption_length = interp_material_property(wavelengths, material.absorption_length)
       #scattering_length = interp_material_property(wavelengths, material.scattering_length)
       #reemission_prob = interp_material_property(wavelengths, material.reemission_prob)
       #reemission_cdf = interp_material_property(wavelengths, material.reemission_cdf)




if __name__ == '__main__':
   pass
   path = sys.argv[1] if len(sys.argv) > 1 else os.environ['DAE_NAME']
   log.info("daeload %s " % path )
   geometry = daeload(path)

   dump_chroma_materials(geometry)







