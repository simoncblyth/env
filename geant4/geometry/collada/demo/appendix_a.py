#!/usr/bin/env python
"""
Hmm, pycollada sees no geometry but blender seems of import OK.
To get it to see the elements had to swap the xmlns (and version)::

    <?xml version="1.0" encoding="utf-8"?>
    <COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
    <!--COLLADA xmlns="http://www.collada.org/2008/03/COLLADASchema" version="1.5.0"-->
     
"""
import os, logging
logging.basicConfig(level=logging.DEBUG)

import lxml
import collada as co

path = os.path.expandvars("$ENV_HOME/graphics/collada/appendix_a.dae")
print path
assert os.path.exists(path), path

dae = co.Collada(path)
print dae
print dae.geometries   # hmm getting none


geom = dae.geometries[0]
print geom 

#print lxml.etree.tostring( dae.xmlnode )



