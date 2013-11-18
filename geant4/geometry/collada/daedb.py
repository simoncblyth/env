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

    dbpath = opts.daedbpath 
    if os.path.exists(dbpath):
        log.info("remove pre-existing db file %s " % dbpath)
        os.remove(dbpath)

    geom_t = Table( dbpath, "geom", idx="int",name="text", nvertex="int", lvid="text", geoid="text" )
    if opts.points:
        point_t = Table( dbpath, "point", id="int",idx="int",x="float",y="float",z="float")

    log.info("building tables for %s nodes " % len(DAENode.registry))
    for node in DAENode.registry:
        id = node.id
        idx = node.index
        lvid = node.lv.id[:-9]  # chop the pointer
        geoid = node.geo.geometry.id[:-9] 
        if idx % 1000 == 0:
            log.info("building tables for node %s %s %s %s " % (idx,id,lvid,geoid))

        prim = list(node.boundgeom.primitives())
        assert len(prim) == 1 , prim
        bpl = list(node.boundgeom.primitives())[0]  
        nvertex = len(bpl.vertex)
        pass
        geom_t.add( idx=idx, name=id, nvertex=nvertex, lvid=lvid, geoid=geoid )
        pass
        if opts.points:
            for pid,xyz in enumerate(bpl.vertex):
                x,y,z = map(float,xyz)
                point_t.add(id=pid,idx=idx,x=x,y=y,z=z)

    pass
    log.info("writing geom_t to %s " % dbpath )
    geom_t.insert()
    if opts.points:
        log.info("writing point_t to %s " % dbpath )
        point_t.insert()
    log.info("completed writing to %s " % dbpath )

if __name__ == '__main__':
    main()


