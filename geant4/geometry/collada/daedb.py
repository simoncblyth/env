#!/usr/bin/env python
"""
DAEDB
=======

Create summary sqlite3 DB of DAE geometry info, for comparison
against the WRL one from VRML2.

::

    [blyth@belle7 ~]$ daedb.py --daepath '$LOCAL_BASE/env/geant4/geometry/gdml/g4_10.dae'
    2013-11-14 19:36:36,534 env.geant4.geometry.collada.daenode INFO     /home/blyth/env/bin/daedb.py
    2013-11-14 19:36:36,534 env.geant4.geometry.collada.daenode INFO     DAENode.parse pycollada parse /data1/env/local/env/geant4/geometry/gdml/g4_10.dae 
    2013-11-14 19:36:38,578 env.geant4.geometry.collada.daenode INFO     pycollada parse completed 
    2013-11-14 19:36:38,891 env.geant4.geometry.collada.daenode INFO     pycollada binding completed, found 12230  
    2013-11-14 19:36:38,891 env.geant4.geometry.collada.daenode INFO     create DAENode heirarchy 
    ...
    2013-11-14 19:36:41,533 env.geant4.geometry.collada.daenode INFO     registry 12230 
    2013-11-14 19:36:41,534 env.geant4.geometry.collada.daenode INFO     lookup 12230 
    2013-11-14 19:36:41,534 env.geant4.geometry.collada.daenode INFO     idlookup 12230 
    2013-11-14 19:36:41,534 env.geant4.geometry.collada.daenode INFO     ids 12230 
    2013-11-14 19:36:41,534 env.geant4.geometry.collada.daenode INFO     rawcount 36690 
    2013-11-14 19:36:41,534 env.geant4.geometry.collada.daenode INFO     created 12230 
    2013-11-14 19:36:41,534 env.geant4.geometry.collada.daenode INFO     root   top.0             -  
    2013-11-14 19:36:41,534 env.geant4.geometry.collada.daenode INFO     index linking DAENode with boundgeom 12230 volumes 
    2013-11-14 19:36:41,557 env.geant4.geometry.collada.daenode INFO     index linking completed
    2013-11-14 19:36:44,549 env.geant4.geometry.collada.daedb INFO     writing to /data1/env/local/env/geant4/geometry/gdml/g4_10.dae.db 


    [blyth@belle7 ~]$ echo "select count(*) from geom ;" | sqlite3 $LOCAL_BASE/env/geant4/geometry/gdml/g4_10.dae.db 
    12230


"""
import sys, os, logging
log = logging.getLogger(__name__)
from env.db.simtab import Table
from daenode import DAENode, parse_args

def main():
    opts, args = parse_args(__doc__) 
    DAENode.parse( opts.daepath )

    geom_t = Table( opts.daedbpath, "geom", idx="int",name="text", nvertex="int", lvid="text", geoid="text" )
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

if __name__ == '__main__':
    main()


