#!/usr/bin/env python

import os, collada, logging
log = logging.getLogger(__name__)

boundgeom = []
nleaf = 0 

def rprint(node, depth=0, index=0 ):
    print "    " * depth, "[%d.%d] %s " % (depth, index, node)
    if not hasattr(node,'children') or len(node.children) == 0:# leaf
        global nleaf
        global boundgeom
        print "leaf", nleaf
        gprint(node) 
        bgprint(boundgeom[nleaf])
        nleaf += 1
    else:
        for index, child in enumerate(node.children):
            rprint(child, depth + 1, index )

def gprint(node):
    """
    When applied to the leaf GeometryNode this gives local coordinates
    """
    for bg in node.objects('geometry'):
        bgprint(bg)

def bgprint(bg):
    print bg 
    for bp in bg.primitives():
        print "max", bp.vertex.max(axis=0)
        print "min", bp.vertex.min(axis=0)
        print "dif", bp.vertex.max(axis=0)-bp.vertex.min(axis=0)



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    dae = collada.Collada("subcopy.dae")
    print dae

    top = dae.scene.nodes[0]

    nleaf = 0 
    boundgeom = list(top.objects('geometry'))
    print "boundgeom", len(boundgeom)
    rprint( top )
    assert len(boundgeom) == nleaf 
