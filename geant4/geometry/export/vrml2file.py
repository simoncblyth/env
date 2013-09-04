#!/usr/bin/env python
"""
VRML2 PARSE AND PERSIST
========================

Parse VRML2 files created by the Geant4 VRML2FILE driver 
and insert the shapes found into an Sqlite3 DB for 
easy inspection.

Quick testing
-------------

For speed only the head of the .wrl is processed::

     ./vrml2file.py --create --quick g4_00.wrl
     ./vrml2file.py --create --quick --extend g4_00.wrl

Full run
----------

Formerly before `point` and `xshape` tables were added the parse and
persist of the 85M wrl took only 10s. After adding `point` and `shape` 
it now takes 8 minutes::

    simon:export blyth$ ./vrml2file.py --create --extend g4_00.wrl
    2013-09-04 18:13:26,551 __main__ INFO     ./vrml2file.py --create --extend g4_00.wrl
    2013-09-04 18:13:26,553 __main__ INFO     create
    2013-09-04 18:14:13,130 __main__ INFO     remove pre-existing db file /Users/blyth/env/geant4/geometry/export/g4_00.db 
    2013-09-04 18:14:13,216 __main__ INFO     gathering geometry 
    2013-09-04 18:16:48,891 __main__ INFO     start persisting to /Users/blyth/env/geant4/geometry/export/g4_00.db 
    2013-09-04 18:21:17,016 __main__ INFO     completed persisting to /Users/blyth/env/geant4/geometry/export/g4_00.db 
    2013-09-04 18:21:18,227 __main__ INFO     extend
    2013-09-04 18:21:18,230 __main__ INFO     drop table if exists xshape 
    2013-09-04 18:21:18,232 __main__ INFO     create table xshape as select sid, count(*) as npo, sum(x) as sumx, avg(x) as ax, min(x) as minx, max(x) as maxx, max(x) - min(x) as dx,sum(y) as sumy, avg(y) as ay, min(y) as miny, max(y) as maxy, max(y) - min(y) as dy,sum(z) as sumz, avg(z) as az, min(z) as minz, max(z) as maxz, max(z) - min(z) as dz from point group by sid 
    simon:export blyth$ 

    simon:export blyth$ du -hs g4_00.*
    128M    g4_00.db
     81M    g4_00.wrl




Inspect Shapes
---------------

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

    simon:export blyth$ echo select src from shape where id=12222 \; | sqlite3 -noheader g4_00.db 
    #---------- SOLID: /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab2.1002
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


Full Overlapping volumes, dodgy dozen
---------------------------------------
::

    sqlite> select count(distinct(src)) from shape ; 
    12223

    simon:export blyth$ echo "select count(distinct(src)) from shape ;" | sqlite3 -noheader g4_00.db 
    12229       

#. 6 more after including the volume name comment metadata first line suggests a small number of absolute position duplicated shapes with different volume names
#. confirmed that assertion using `shape.hash` digest that excludes the name metadata 

The dodgy dozen, six pairs of volumes are precisely co-located::

    sqlite> select hash, group_concat(name), group_concat(id)  from shape group by hash having count(*) > 1 ;
    hash                              group_concat(name)                                                                                                                           group_concat(id)
    --------------------------------  ---------------------------------------------------------------------------------------------                                                ----------------
    036f14cfb2e7bbe62226d213bd3e7780  /dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000,/dd/Geometry/CalibrationSources/lvMainSSCavity#pvAmCCo60SourceAcrylic.1000  6400,6401       
    2043a400a35f062979ddfa73254cac9d  /dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000,/dd/Geometry/CalibrationSources/lvMainSSCavity#pvAmCCo60SourceAcrylic.1000  6318,6319       
    547dd4e8ad4c711815456951753d8fa9  /dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000,/dd/Geometry/CalibrationSources/lvMainSSCavity#pvAmCCo60SourceAcrylic.1000  4570,4571       
    b7e229d741481e47f3c06236dbc2961d  /dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000,/dd/Geometry/CalibrationSources/lvMainSSCavity#pvAmCCo60SourceAcrylic.1000  6230,6231       
    be270355bc36384aa290479074aaec4e  /dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000,/dd/Geometry/CalibrationSources/lvMainSSCavity#pvAmCCo60SourceAcrylic.1000  4658,4659       
    c35f0b07cfa25126ec1b156aca3364d8  /dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000,/dd/Geometry/CalibrationSources/lvMainSSCavity#pvAmCCo60SourceAcrylic.1000  4740,4741       
    sqlite> 


    sqlite> select substr(src,0,600) from shape where id = 6401 ;
    #---------- SOLID: /dd/Geometry/CalibrationSources/lvMainSSCavity#pvAmCCo60SourceAcrylic.1000
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
                                            -15954.9 -805788 -4145.32,
                                            -15953.4 -805788 -4145.32,
                                            -15951.7 -805789 -4145.32,
                                            -15950 -805789 -4145.32,
                                            -15948.4 -805788 -4145.32,
                                            -15946.9 -805787 -4145.32,
                                            -15945.8 -805786 -4145.32,
                                            -15945 -805784 -4145.32,
                                            -15944.6 -805783 -4145.32,
                                            -15944.7 -805781 -4145.32,
                                            -15945.3 -8
    sqlite> 
    sqlite> 
    sqlite> 
    sqlite> select substr(src,0,600) from shape where id = 6400 ;
    #---------- SOLID: /dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000
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
                                            -15954.9 -805788 -4145.32,
                                            -15953.4 -805788 -4145.32,
                                            -15951.7 -805789 -4145.32,
                                            -15950 -805789 -4145.32,
                                            -15948.4 -805788 -4145.32,
                                            -15946.9 -805787 -4145.32,
                                            -15945.8 -805786 -4145.32,
                                            -15945 -805784 -4145.32,
                                            -15944.6 -805783 -4145.32,
                                            -15944.7 -805781 -4145.32,
                                            -15945.3 -805779 -414





Small number of different src lengths
----------------------------------------

Only ~53 different lengths of src but 12k distinct src. 
Small number of shapes are repeated in different positions, eg PMT rotations.


::

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




Shape extents
--------------

::

    sqlite> create table xshape as select sid, count(*) as npo,  max(x)-min(x) as dx, max(y)-min(y) as dy, max(z)-min(z) as dz, avg(x) as ax, avg(y) as ay,  avg(z) as az from point group by sid limit 10 ;
    sqlite> 
    sqlite> select * from xshape ;
    sid         npo         dx          dy          dz          ax          ay          az        
    ----------  ----------  ----------  ----------  ----------  ----------  ----------  ----------
    1           8           69139.8     69140.0     37994.2     -16519.99   -802110.0   3892.9    
    2           16          36494.56    45091.0     15000.29    -11482.888  -808975.25  2639.855  
    3           40          13823.07    15602.0     44.0        -15876.303  -803178.07  -2088.0   
    4           8           3019.4      3010.0      78.0        -11612.4    -799007.25  683.904   
    5           8           2911.3      2911.0      75.0        -11611.25   -799018.5   683.904   
    6           8           2897.5      2897.0      6.0         -11611.25   -799018.5   669.904   
    7           8           2869.9      2870.0      2.0         -11611.25   -799018.25  669.904   
    8           8           1896.7      1332.0      2.0         -11124.675  -799787.25  669.904   
    9           8           1896.6      1332.0      2.0         -11263.7    -799567.75  669.904   
    10          8           1896.7      1332.0      2.0         -11402.725  -799348.0   669.904   

    sqlite> create table xshape as select sid, count(*) as npo,  min(x) as minx, max(x) as maxx, max(x)-min(x) as dx, min(y) as miny, max(y) as maxy, max(y)-min(y) as dy, min(z) as minz, max(z) as max(z), max(z)-min(z) as dz, avg(x) as ax, avg(y) as ay,  avg(z) as az from point group by sid ;
    Error: table xshape already exists
    sqlite> drop table xshape ;
    sqlite> create table xshape as select sid, count(*) as npo,  max(x)-min(x) as dx, max(y)-min(y) as dy, max(z)-min(z) as dz, avg(x) as ax, avg(y) as ay,  avg(z) as az from point group by sid ;
    sqlite> select count(*) from xshape ;
    count(*)  
    ----------
    12229     



"""
import os, sys, logging
from env.db.simtab import Table
from md5 import md5
log = logging.getLogger(__name__) 


class WRLRegion(object):
   pfx_point_region = 'point ['
   pfx_coordIndex_region = 'coordIndex ['
   pfx_close_region = ']'
   def __init__(self, src, name=None, indx=None):
        self.src = src[:] 
        self.hash = md5("".join(src[1:])).hexdigest()
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
                      xyz = map(float,s[:-1].split(" "))
                      self.point.append(xyz) 
                      #print "%s : %s " % ( region, xyz ) 
                  elif region == "coordIndex":
                      assert s[-1] == ",", s
                      cdx = s[:-1].split(", ") 
                      #print "%s : %s " % ( region , cdx )


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
        """
        The token lines are used to mark the 
        end of the prior region, and the start of a new one. 
        """  
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
            pass
        self.buffer.append(line) 

    def save(self, path ):
        path = os.path.abspath(path)
        if os.path.exists(path):
            log.info("remove pre-existing db file %s " % path)
            os.remove(path)
        pass

        shape = Table(path, "shape", id="int",name="text", src="blob", len="int", hash="text")
        point = Table(path, "point", id="int",sid="int",x="float",y="float",z="float")

        log.info("gathering geometry ")  
        for sh in self:
            if sh.name == 'camera':
                pass
            else:
                sid = sh.indx
                shape.add(id=sid,name=sh.name, src="".join(sh.src), len=len(sh.src), hash=sh.hash )
                sh()
                for pid,(x,y,z) in enumerate(sh.point):
                    point.add(id=pid,sid=sid,x=x,y=y,z=z)
            pass
        # writes to the DB a table at a time
        log.info("start persisting to %s " % path ) 
        shape.insert()   
        point.insert()   
        log.info("completed persisting to %s " % path ) 


    def extend(self, path, tn="xshape"):
        dummy = Table(path)
        sammd_ = lambda _:"sum(%(_)s) as sum%(_)s, avg(%(_)s) as a%(_)s, min(%(_)s) as min%(_)s, max(%(_)s) as max%(_)s, max(%(_)s) - min(%(_)s) as d%(_)s" % locals() 
        xsql = "select sid, count(*) as npo, " + ",".join(map(sammd_, ("x","y","z"))) + " from point group by sid "
        sqls = ["drop table if exists %(tn)s " % locals(),
                "create table %(tn)s as " % locals() + xsql ]
        for sql in sqls:        
            log.info(sql)
            for ret in dummy(sql):
                log.info(ret) 


    def dump(self):
        for sh in wrlp:
            print repr(sh)


def parse_args(doc):
    """
    Return config dict and commandline arguments 

    :param doc:
    :return: cnf, args  
    """
    from optparse import OptionParser
    op = OptionParser(usage=doc)
    op.add_option("-o", "--logpath", default=None )
    op.add_option("-l", "--loglevel",   default="INFO", help="logging level : INFO, WARN, DEBUG ... Default %default"  )
    op.add_option("-f", "--logformat", default="%(asctime)s %(name)s %(levelname)-8s %(message)s" )
    op.add_option("-q", "--quick", action="store_true", help="Quick run just on the head of the input file.", default=False )
    op.add_option("-c", "--create", action="store_true", help="Create the DB from the source wrl.", default=False )
    op.add_option("-x", "--extend", action="store_true", help="Create the extents table from the pre-created DB.", default=False )
    opts, args = op.parse_args()
    level = getattr( logging, opts.loglevel.upper() )

    if opts.logpath:  # logs to file as well as console, needs py2.4 + (?)
        logging.basicConfig(format=opts.logformat,level=level,filename=opts.logpath)
        console = logging.StreamHandler()
        console.setLevel(level)
        formatter = logging.Formatter(opts.logformat)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)  # add the handler to the root logger
    else:
        try: 
            logging.basicConfig(format=opts.logformat,level=level)
        except TypeError:
            hdlr = logging.StreamHandler()              # py2.3 has unusable basicConfig that takes no arguments
            formatter = logging.Formatter(opts.logformat)
            hdlr.setFormatter(formatter)
            log.addHandler(hdlr)
            log.setLevel(level)
        pass
    pass

    log.info(" ".join(sys.argv))
    #logging.getLogger().setLevel(loglevel)
    return opts, args





if __name__ == '__main__':
    opts, args = parse_args(__doc__)
    path = args[0]
    base, ext = os.path.splitext(path)
    dbpath = os.path.abspath(base + ".db")

    wrlp = WRLParser()

    if opts.create:
        log.info("create") 
        if opts.quick:
            wrlp(path=None,cmd="head -226 %(path)s " % locals()) # head only testing, fine tuned to avoid shapes without any points arising from truncation 
        else:      
            wrlp(path)
        pass     
        wrlp.save(dbpath)
    else:
        log.info("skip create") 


    if opts.extend:
        log.info("extend") 
        wrlp.extend(dbpath, "xshape")
    else:    
        log.info("skip extend") 

