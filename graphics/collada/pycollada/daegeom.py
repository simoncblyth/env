#!/usr/bin/env python
"""

::

     collada-;collada-cd
     daegeom.py 3199.dae 0

     daegeom.py $LOCAL_BASE/env/geant4/geometry/xdae/g4_01.dae 1
     daegeom.py $LOCAL_BASE/env/graphics/collada/0.dae 1

"""

import os, logging, sys
log = logging.getLogger(__name__)

from env.geant4.geometry.vrml2.vrml2db import VRML2DB
import collada
import numpy

def primfix(self):
    """
    Original pycollada uses post multiplication for lineset,polylist,triangleset::
 
        self._vertex = None if pl._vertex is None else numpy.asarray(pl._vertex * M[:3,:3]) + matrix[:3,3]
        self._normal = None if pl._normal is None else numpy.asarray(pl._normal * M[:3,:3])

    This works well for volume 1, but not below.
    """
    assert self.__class__.__name__ in ('BoundLineSet','BoundPolylist','BoundTriangleSet'), self
    M = numpy.asmatrix(self.matrix).transpose()
    if self.original._vertex is None:
        self._vertex = None 
    else: 
        self._vertex = numpy.asarray(( M[:3,:3] * self.original._vertex.T ).T ) + self.matrix[:3,3]

    if self.original._normal is None:
        self._normal = None 
    else: 
        self._normal = numpy.asarray(( M[:3,:3] * self.original._normal.T ).T ) 



def dump_geom(path, index):
    db = VRML2DB()
    vpo = db.points(index)    

    dae = collada.Collada(path)
    top = dae.scene.nodes[0]
    log.info("dump_geom from %s boundgeom index %s " % (path, index))
    boundgeom = list(top.objects('geometry'))
    bg = boundgeom[int(index)]
    prim = list(bg.primitives())
    assert len(prim) == 1, len(prim)
    bp = prim[0]
    bpl = list(bg.primitives())[0] 

    print "before primfix", bpl, "nvtx:", len(bpl.vertex)
    print bpl.vertex

    primfix(bpl)
    print "after primfix", bpl, "nvtx:", len(bpl.vertex)
    print bpl.vertex

    print "from VRML2DB: \n", vpo

    #for i, po in enumerate(bpl):
    #    print i, po, po.indices



def main():
    logging.basicConfig(level=logging.INFO)
    dump_geom(*sys.argv[1:])


if __name__ == '__main__':
    main()


