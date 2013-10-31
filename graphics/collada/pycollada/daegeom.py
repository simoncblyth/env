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
    **RENDERED OBSOLETE BY _MONKEY_MATRIX_LOAD**

    Original pycollada lineset,polylist,triangleset
    use post multiplication by the transposed rotation portion of the node matrix and 
    broadcast addition of the translation portion::

        M = numpy.asmatrix(matrix).transpose()
        self._vertex = None if pl._vertex is None else numpy.asarray(pl._vertex * M[:3,:3]) + matrix[:3,3]
        self._normal = None if pl._normal is None else numpy.asarray(pl._normal * M[:3,:3])

    Easier to stay using post-multiplication to be consistent with the recursive transformations, 
    but not to transpose the rotation matrix (same as invert for a rotation matrix)
    like pycollada does (or store the transposed to avoid 

    Initial fix used::

        M = numpy.asmatrix(self.matrix).transpose()
        self._vertex = numpy.asarray(( M[:3,:3] * self.original._vertex.T ).T ) + self.matrix[:3,3]
        self._normal = numpy.asarray(( M[:3,:3] * self.original._normal.T ).T ) 

    But can avoid all the transposing by post-multiplying the untransposed original matrix. Using 
    numpy.dot avoids conversion from numpy array to matrix.

    To avoid changing pycollada here and in the recursive transformations
    could change the collada file to store the transposed rotation 
    and the translation as is.

    This works for PV1, but not below.
    """
    assert self.__class__.__name__ in ('BoundLineSet','BoundPolylist','BoundTriangleSet'), self
    if self.original._vertex is None:
        self._vertex = None 
    else: 
        self._vertex = numpy.dot(self.original._vertex,self.matrix[:3,:3]) + self.matrix[:3,3]

    if self.original._normal is None:
        self._normal = None 
    else: 
        self._normal = numpy.dot(self.original._normal, self.matrix[:3,:3]) 


def _monkey_matrix_load(_collada,node, diddle=True):
    """
    Avoid changing pycollada in multiple places by monkey patching 
    just the matrix loading to diddle the matrix. 
    The matrix diddling just inverts the rotation portion.

    After doing this it is wring to do the primfix too.

    The advantage over primfix, is that this way also works 
    appropriately with the recursive Node transformations.

    Could avoid having to use this monkeypatch by doing this diddle
    within G4DAEWrite, ie writing diddled to the .dae
    """
    floats = numpy.fromstring(node.text, dtype=numpy.float32, sep=' ')

    if diddle:
        original = floats.copy()
        original.shape = (4,4)
        matrix = numpy.identity(4)
        matrix[:3,:3] = original[:3,:3].T   # transpose/invert the 3x3 rotation portion
        matrix[:3,3] = original[:3,3]       # tack back the translation
        floats = matrix.ravel()

    return collada.scene.MatrixTransform(floats, node)

collada.scene.MatrixTransform.load = staticmethod(_monkey_matrix_load)


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

    print "bpl", bpl, "nvtx:", len(bpl.vertex)
    print bpl.vertex
    for i, po in enumerate(bpl):
        print i, po, po.indices

    #  primfix not needed when using monkey patched matrix loading 
    #primfix(bpl)
    #print "after primfix", bpl, "nvtx:", len(bpl.vertex)
    #print bpl.vertex

    print "from VRML2DB: %s \n" % len(vpo), vpo


def main():
    logging.basicConfig(level=logging.INFO)
    dump_geom(*sys.argv[1:])


if __name__ == '__main__':
    main()


