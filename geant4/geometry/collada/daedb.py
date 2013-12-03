#!/usr/bin/env python
"""
DAEDB
=======

Create summary sqlite3 DB of DAE geometry info, for comparison
against the WRL one from VRML2.

Straightforward write to DB once at the end, takes 5 min::

    [blyth@belle7 ~]$ daedb.py --daepath '$LOCAL_BASE/env/geant4/geometry/daeserver/VDGX_20131121-2043_g4_00.dae'       
    2013-12-03 15:34:54,665 env.geant4.geometry.collada.daenode INFO     /home/blyth/env/bin/daedb.py
    ...
    2013-12-03 15:37:03,314 env.geant4.geometry.collada.daedb INFO     perform final DB insert for inode 12229 insertsize 0 
    2013-12-03 15:37:03,315 env.geant4.geometry.collada.daedb INFO     writing tables to /data1/env/local/env/geant4/geometry/daeserver/VDGX_20131121-2043_g4_00.dae.db 
    2013-12-03 15:37:03,315 env.geant4.geometry.collada.daedb INFO     writing geom_t to /data1/env/local/env/geant4/geometry/daeserver/VDGX_20131121-2043_g4_00.dae.db 
    2013-12-03 15:37:03,824 env.geant4.geometry.collada.daedb INFO     writing point_t to /data1/env/local/env/geant4/geometry/daeserver/VDGX_20131121-2043_g4_00.dae.db 
    2013-12-03 15:38:20,287 env.geant4.geometry.collada.daedb INFO     writing face_t to /data1/env/local/env/geant4/geometry/daeserver/VDGX_20131121-2043_g4_00.dae.db 
    2013-12-03 15:40:05,296 env.geant4.geometry.collada.daedb INFO     completed writing to /data1/env/local/env/geant4/geometry/daeserver/VDGX_20131121-2043_g4_00.dae.db 

Specifying a non-zero insertsize prevents holding everything in memory, 
instead the DB writes are done in chunks corresponding to the entries for the specified numbers of volumes.

    [blyth@belle7 ~]$ daedb.py --daepath '$LOCAL_BASE/env/geant4/geometry/daeserver/VDGX_20131121-2043_g4_00.dae'   --insertsize 1000   

For examples of queries see ~/env/geant4/geometry/collada/dae_cf_wrl.rst


"""
import sys, os, logging
log = logging.getLogger(__name__)
from env.db.simtab import Table
from daenode import DAENode, parse_args


class DAEDB(object):
    def __init__(self, dbpath, opts):
        if os.path.exists(dbpath):
            log.info("remove pre-existing db file %s " % dbpath)
            os.remove(dbpath)
        self.dbpath = dbpath
        self.opts = opts
        pass
        geom_t = Table( dbpath, "geom", idx="int",name="text", nvertex="int", nface="int", lvid="text", geoid="text" )
        point_t = None
        face_t = None
        pass
        if opts.points:
            point_t = Table( dbpath, "point", id="int",idx="int",x="float",y="float",z="float")
        if opts.faces:
            face_t = Table( dbpath, "face", id="int",idx="int",v0="int",v1="int",v2="int", v3="int", vx="text", nv="int" , _big=True )
        pass    
        self.geom_t = geom_t
        self.point_t = point_t
        self.face_t = face_t
        
    def __call__(self):
        log.info("building tables for %s nodes " % len(DAENode.registry))
        insertsize = self.opts.insertsize 
        for inode, node in enumerate(DAENode.registry):
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
            nface = bpl.npolygons
            pass
            self.geom_t.add( idx=idx, name=id, nvertex=nvertex, nface=nface, lvid=lvid, geoid=geoid )
            pass
            if not self.point_t is None:
                for pid,xyz in enumerate(bpl.vertex):
                    x,y,z = map(float,xyz)
                    self.point_t.add(id=pid,idx=idx,x=x,y=y,z=z)
            pass        
            if not self.face_t is None:
                for fid,f in enumerate(bpl.polygons()):
                    ii = f.indices.tolist()
                    nv = len(ii)
                    assert nv in (3,4), (nv, "unexpected number of vertices for face ")
                    if nv == 3:
                        ii.append(-1)
                    self.face_t.add(id=fid,idx=idx,v0=ii[0],v1=ii[1],v2=ii[2],v3=ii[3],vx=",".join(map(str,ii)), nv=nv)
            pass        
            if insertsize > 0 and inode % insertsize == 0:
                log.info("perform DB insert for inode %s insertsize %s " % (inode, insertsize ))
                self.insert()
        pass
        log.info("perform final DB insert for inode %s insertsize %s " % (inode, insertsize ))
        self.insert()

    def insert(self, clear=True):    
        """
        Adding to a simtab table collects the kwa into a list.
        These are then written to sqlite by the insert.  Specifying
        ``clear=True`` wipes the list after doing the DB write and
        is what you normally want.
        """
        dbpath = self.dbpath
        log.info("writing tables to %s " % dbpath )
        if self.geom_t:
            log.info("writing geom_t to %s " % dbpath )
            self.geom_t.insert(clear=clear)
        if self.point_t:
            log.info("writing point_t to %s " % dbpath )
            self.point_t.insert(clear=clear)
        if self.face_t:
            log.info("writing face_t to %s " % dbpath )
            self.face_t.insert(clear=clear)
        log.info("completed writing to %s " % dbpath )


def main():
    opts, args = parse_args(__doc__) 
    DAENode.parse( opts.daepath )
    db = DAEDB( opts.daedbpath, opts  )
    db()
    pass

if __name__ == '__main__':
    main()


