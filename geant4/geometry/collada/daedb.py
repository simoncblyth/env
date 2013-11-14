#!/usr/bin/env python
"""
DAEDB
=======

Create summary sqlite3 DB of DAE geometry info, for comparison
against the WRL one from VRML2.

"""
import os, logging
log = logging.getLogger(__name__)

from env.db.simtab import Table

if __name__ == '__main__':
    pass
    logging.basicConfig(level=logging.INFO)
    from daenode import DAENode, Defaults
    DAENode.parse(Defaults.daepath)

    geom_t = Table(Defaults.dbpath, "geom", idx="int",name="text", nvertex="int", lvid="text", geoid="text" )
    for node in DAENode.registry:
        id = node.id
        prim = list(node.boundgeom.primitives())
        assert len(prim) == 1 , prim
        bpl = list(node.boundgeom.primitives())[0]  
        nvertex = len(bpl.vertex)
        lvid = node.lv.id[:-9]  # chop the pointer
        geoid = node.geo.geometry.id[:-9]  
        geom_t.add( idx=node.index, name=node.id, nvertex=nvertex, lvid=lvid, geoid=geoid )
    pass
    log.info("writing to %s " % geom_t.path )
    geom_t.insert()




