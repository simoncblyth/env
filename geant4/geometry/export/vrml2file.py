#!/usr/bin/env python
"""
VRML2 PARSE AND PERSIST
========================

Parse VRML2 files created by the Geant4 VRML2FILE driver 
and insert the shapes found into an Sqlite3 DB for 
easy inspection.

Usage::

    [blyth@belle7 export]$ time ./vrml2file.py g4_00.wrl
    INFO:__main__:remove pre-existing db file /home/blyth/env/geant4/geometry/export/g4_00.db 
    INFO:__main__:saving geometry into file /home/blyth/env/geant4/geometry/export/g4_00.db 

    real    0m9.140s
    user    0m6.706s
    sys     0m1.005s

Takes about 10s to parse and persist an 85M WRL::

    blyth@belle7 export]$ du -h g4_00.*
    85M     g4_00.db
    82M     g4_00.wrl


TODO
-----

#. deeper parsing and persisting to pull out the coordinates, 
#. allow dynamic repositioning of shapes to the origin, blender having trouble with large coordinates

Inspect Shapes
---------------

Everything is white with transparency 0.7::

    sqlite> select distinct(substr(src,60,21)) from shape ;
                            diffuseColor 1 1 1

    sqlite> select distinct(substr(src,82,20)) from shape ;
                                    transparency 0.7

Heads of all shapes are identical::

    sqlite> select distinct(substr(src,0,178)) from shape ;
            Shape {
                    appearance Appearance {
                            material Material {
                                    diffuseColor 1 1 1
                                    transparency 0.7
                            }
                    }
                    geometry IndexedFaceSet {
                            coord Coordinate {
                                    point [


::

    echo select \* from shape \; | sqlite3 g4_00.db


    [blyth@belle7 export]$ echo select name, src from shape where indx=12222 \; | sqlite3 g4_00.db 
    /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab2.1002|    Shape {
                    appearance Appearance {
                            material Material {
                                    diffuseColor 1 1 1
                                    transparency 0.7
                            }
                    }
                    geometry IndexedFaceSet {
                            coord Coordinate {
                                    point [
                                            -22540.9 -796477 -12260,
                                            -22834.2 -796414 -12260,
                                            -23724.9 -800569 -12260,
                                            -23431.5 -800632 -12260,
                                            -22540.9 -796477 -2260,
                                            -22834.2 -796414 -2260,
                                            -23724.9 -800569 -2260,
                                            -23431.5 -800632 -2260,
                                    ]
                            }
                            coordIndex [
                                    0, 3, 2, 1, -1,
                                    4, 7, 3, 0, -1,
                                    7, 6, 2, 3, -1,
                                    6, 5, 1, 2, -1,
                                    5, 4, 0, 1, -1,
                                    4, 5, 6, 7, -1,
                            ]
                            solid FALSE
                    }
            }


Only ~53 different lengths of src but 12k distinct src. 
Small number of shapes are repeated in different positions, eg PMT rotations.

::

    sqlite> select count(distinct(src)) from shape ; 
    12223

    sqlite> select len,count(*) as N from shape group by len order by len ;
    31|5362
    36|1
    45|163
    47|160
    52|1
    ...
    859|672
    892|6
    941|64
    961|2
    979|672
    1031|2
    1291|672
    1588|6
    1707|2
    1869|2







"""
import os, sys, logging
from env.db.simtab import Table
log = logging.getLogger(__name__) 


class WRLRegion(object):
   pfx_point_region = 'point ['
   pfx_coordIndex_region = 'coordIndex ['
   pfx_close_region = ']'
   def __init__(self, src, name=None, indx=None):
        self.src = src[:] 
        self.name = name
        self.indx = indx
        self.point = []
        self.coordIndex = []

   def __repr__(self):
        return "# [%-6s] (%10s) :  %s " % ( self.indx, len(self.src), self.name )
   def __str__(self):
        return "\n".join( [repr(self), "".join(self.src)] ) 

   def __call__(self):
        token, region = False, None
        for line in self.src:
             s = line.lstrip().strip()
             if s == self.pfx_point_region:
                  token, region = True, "point"
             elif s == self.pfx_coordIndex_region:
                  token, region = True, "coordIndex"
             elif s == self.pfx_close_region:
                  token, region = True, None
             else:
                  token = False

             if not token:
                  if region == "point":
                      assert s[-1] == ",", s
                      xyz = s[:-1].split(" ") 
                      print "%s : %s " % ( region, xyz ) 
                  elif region == "coordIndex":
                      assert s[-1] == ",", s
                      cdx = s[:-1].split(", ") 
                      print "%s : %s " % ( region , cdx )


class WRLParser(list):
    pfx_camera = '#---------- CAMERA'
    pfx_solid = '#---------- SOLID: '
    def __init__(self): 
        self.region = None
        self.lpfx_solid = len(self.pfx_solid)
        self.lpfx_camera = len(self.pfx_camera)
        self.buffer = []

    def __call__(self, path=None, cmd=None):
        if not path is None:
            fp = open(path)   
        elif not cmd is None:
            fp = os.popen(cmd)
        else:
            assert 0, (path, cmd)

        for line in fp.readlines():
            self.parse_line(line)
            pass
        self._add_region()  # for the last region 

    def _add_region(self):
        if not self.region is None:
            reg = WRLRegion( self.buffer, self.region, indx=len(self))  
            self.append( reg )
            self.buffer[:] = []

    def parse_line(self, line):
        token = False 
        if line[0:self.lpfx_camera] == self.pfx_camera:
            token, name = True, 'camera'
        elif line[0:self.lpfx_solid] == self.pfx_solid:
            token, name = True, line[self.lpfx_solid:-1]
        else:
            pass

        if token: 
            self._add_region()             
            self.region = name 
        else:
            self.buffer.append(line) 

    def save(self, path ):
        path = os.path.abspath(path)
        if os.path.exists(path):
            log.info("remove pre-existing db file %s " % path)
            os.remove(path)
        pass
        log.info("saving geometry into file %s " % path )  

        schema = dict(name="text", src="blob", len="int", indx="int" )

        tab = Table(path, "shape", **schema)
        for sh in self:
            if sh.name == 'camera':
                pass
            else:
                rec = dict(name=sh.name, indx=sh.indx, src="".join(sh.src), len=len(sh.src))
                tab.add(**rec)

                sh()

            pass
        tab.insert()   # writes to the DB all at once

    def dump(self):
        for sh in wrlp:
            print repr(sh)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    wrlp = WRLParser()

    path = sys.argv[1]

    wrlp(path=None,cmd="head -100 %(path)s " % locals()) # head only testing
    #wrlp(path)

    base, ext = os.path.splitext(path)
    dbpath = base + ".db"
    wrlp.save(dbpath)



