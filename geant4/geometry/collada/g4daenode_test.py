#!/usr/bin/env python
"""

From bash::

   ./daenode_test.py /usr/local/env/geant4/geometry/gdml/VDGX_20131121-1957/g4_00.dae

From IPython::

   In [1]: run daenode_test.py /usr/local/env/geant4/geometry/gdml/VDGX_20131121-1957/g4_00.dae


"""
import sys, logging
import lxml.etree as ET
log = logging.getLogger(__name__)

from collada.xmlutil import COLLADA_NS as NS
from env.geant4.geometry.collada.g4daenode import DAENode

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    DAENode.parse( sys.argv[1] )   
    DAENode.summary()

    for node in DAENode.registry:
        nom = node.metadata
        bgm = node.boundgeom_metadata()
        cout = bgm.get('cout',None)
        cerr = bgm.get('cerr',None)
        smry = nom.get('polysmry', None)
        print node.index, smry 






