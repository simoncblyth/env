#!/usr/bin/env python
"""

::

     collada-;collada-cd
     daegeom.py 3199.dae 0

"""

import os, logging, sys
log = logging.getLogger(__name__)
import collada

def dump_geom(path, index):
    dae = collada.Collada(path)
    top = dae.scene.nodes[0]
    log.info("dump_geom from %s boundgeom index %s " % (path, index))
    boundgeom = list(top.objects('geometry'))
    bg = boundgeom[int(index)]
    prim = list(bg.primitives())
    assert len(prim) == 1, len(prim)
    bp = prim[0]
    for po in list(bp.polygons()):
        print po.vertices, po.indices

def main():
    logging.basicConfig(level=logging.INFO)
    dump_geom(*sys.argv[1:])


if __name__ == '__main__':
    main()


