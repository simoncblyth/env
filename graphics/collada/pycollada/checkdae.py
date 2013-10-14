#!/usr/bin/env python
"""

scp N:env/geant4/geometry/xdae/test.dae .

::


Observations
-------------

#. transformation matrices living on the anonymous nodes


"""
import logging
log = logging.getLogger(__name__)
import collada as co
import monkey_collada


class Traverse(object):
    def __init__(self, dae):
        self.dae = dae
        self.count = 0 
        self.other = 0 
        self.anon = 0 

    def visit_geo(self, node, depth):
        print depth, node
    def visit(self, node, depth):
        self.count += 1
        id = node.id
        if id is None:
            pass
            print node.matrix 
            self.anon += 1 
        elif 'PMT' in id or 'PoolDetails' in id:
            pass
        else:
            print self.count, depth, label
    def recurse(self, node, depth):
        if hasattr(node,'children') and hasattr(node,'id'):
            self.visit(node, depth)
            for child in node.children:
                self.recurse(child, depth+1)
        else:
            assert node.__class__.__name__ == 'GeometryNode', node
            self.visit_geo(node, depth)
            self.other += 1
    def __call__(self):
        for top in self.dae.scene.nodes:
            self.recurse(top, 0)
    def __str__(self):
        return "%s count %s other (GeomNode) %s anon %s  " % ( self.__class__.__name__, self.count, self.other, self.anon )


def object_count(dae):
    """
    Pre-cooked recursion and application of transform matrices 
    Problem is this looses access to the mother material ?
    """
    npol = 0 
    ntri = 0
    nvtx = 0
    boundgeom = list(dae.scene.objects('geometry'))
    for bg in boundgeom:
        for bp in bg.primitives():
            print bp
            assert bp.__class__.__name__ == 'BoundPolygons'
            npol += len(bp)
            tris = bp.triangleset()
            ntri += len(tris)
            nvtx += len(bp.vertex)
    pass
    print "npol", npol
    print "ntri", ntri
    print "nvtx", nvtx


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    dae = co.Collada("test.dae")
    print dae
    object_count(dae)
    #t = Traverse(dae)
    #t()
    #print t 





