#!/usr/bin/env python
"""

scp N:env/geant4/geometry/xdae/test.dae .

::

    npol 1824947
    ntri 2483650
    nvtx 1264049


Observations
-------------

#. transformation matrices living on the anonymous nodes


TODO

* check the matrix transforms, by comparison against VRML2 output 

::

    In [85]: bp.vertex.min(axis=0) - bp.vertex.min(axis=0)
    Out[85]: array([ 0.,  0.,  0.], dtype=float32)

    In [86]: bp.vertex.min(axis=0) - bp.vertex.max(axis=0)
    Out[86]: array([-13823.15625, -15602.0625 ,   -300.     ], dtype=float32)



Can I recurse manually, in order to access the mother material,  
but still use the BoundGeometry transformation calcs ?


"""
import collada as co
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
    dae = co.Collada("test.dae")
    print dae
    t = Traverse(dae)
    t()
    print t 





