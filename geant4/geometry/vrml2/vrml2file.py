#!/usr/bin/env python
"""
VRML2 PARSE AND PERSIST
========================

Parse VRML2 files created by the Geant4 VRML2FILE driver 
and insert the shapes found into an Sqlite3 DB to provide

#. easy inspection.
#. generation of sub-geometry .wrl files selecting particular volumes

Quick testing
-------------

For speed only the head of the .wrl is processed::

    vrml2file.py --head 124 --save g4_00.wrl -l debug
    vrml2file.py --head 124 --save g4_00.wrl 
    vrml2file.py --save --head 226 --extend g4_01.wrl


Full run but skip shape table for speed and slim DB
------------------------------------------------------

::

    vrml2file.py --save --noshape g4_00.wrl      


Refactored Full run
---------------------

Attempting to do the time consuming things here, to make subsequent shape querying faster.
As this initial parse is done infrequently::

    simon:export blyth$ ./vrml2file.py -cx g4_01.wrl 
    2013-09-12 13:36:31,755 __main__ INFO     ./vrml2file.py -cx g4_01.wrl
    2013-09-12 13:36:31,758 __main__ INFO     create
    2013-09-12 13:37:34,184 __main__ INFO     remove pre-existing db file /Users/blyth/env/geant4/geometry/export/g4_01.db 
    2013-09-12 13:37:34,207 __main__ INFO     gathering geometry, using idoffset 1 
    2013-09-12 13:40:44,221 __main__ INFO     start persisting to /Users/blyth/env/geant4/geometry/export/g4_01.db 
    2013-09-12 13:47:22,216 __main__ INFO     completed persisting to /Users/blyth/env/geant4/geometry/export/g4_01.db 
    2013-09-12 13:47:23,402 __main__ INFO     extend
    2013-09-12 13:47:23,405 __main__ INFO     drop table if exists xshape 
    2013-09-12 13:47:23,441 __main__ INFO     create table xshape as select sid, count(*) as npo, sum(x) as sumx, avg(x) as ax, min(x) as minx, max(x) as maxx, max(x) - min(x) as dx,sum(y) as sumy, avg(y) as ay, min(y) as miny, max(y) as maxy, max(y) - min(y) as dy,sum(z) as sumz, avg(z) as az, min(z) as minz, max(z) as maxz, max(z) - min(z) as dz ,name from point join shape on point.sid = shape.id group by sid 
    simon:export blyth$ 
    simon:export blyth$ du -hs g4_01.*
    253M    g4_01.db
     81M    g4_01.wrl

Only a couple of minutes on N::

    [blyth@belle7 export]$ ./vrml2file.py -cx g4_01.wrl 
    2013-09-16 12:44:09,184 __main__ INFO     ./vrml2file.py -cx g4_01.wrl
    2013-09-16 12:44:09,184 __main__ INFO     create
    2013-09-16 12:44:21,066 __main__ INFO     remove pre-existing db file /home/blyth/env/geant4/geometry/export/g4_01.db 
    2013-09-16 12:44:21,129 __main__ INFO     gathering geometry, using idoffset True idlabel 1 
    2013-09-16 12:44:47,263 __main__ INFO     start persisting to /home/blyth/env/geant4/geometry/export/g4_01.db 
    2013-09-16 12:45:34,089 __main__ INFO     completed persisting to /home/blyth/env/geant4/geometry/export/g4_01.db 
    2013-09-16 12:45:34,260 __main__ INFO     extend
    2013-09-16 12:45:34,265 __main__ INFO     drop table if exists xshape 
    2013-09-16 12:45:34,265 __main__ INFO     create table xshape as select sid, count(*) as npo, sum(x) as sumx, avg(x) as ax, min(x) as minx, max(x) as maxx, max(x) - min(x) as dx,sum(y) as sumy, avg(y) as ay, min(y) as miny, max(y) as maxy, max(y) - min(y) as dy,sum(z) as sumz, avg(z) as az, min(z) as minz, max(z) as maxz, max(z) - min(z) as dz ,name from point join shape on point.sid = shape.id group by sid 

    [blyth@belle7 export]$ du -hs g4_01.*
    254M    g4_01.db
    20K     g4_01.quick.db
    82M     g4_01.wrl


Prior Full run
---------------

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

Dev
----

Make volume construction by query simple, to give optimisation possibilities.
Ordinary newlines not working, so use raw ascii 10::

    simon:export blyth$ echo "select src_head||x'0A'||src_points||x'0A'||src_tail from shape where id=1 ;" | sqlite3 -noheader g4_01.db > a.wrl
    simon:export blyth$ echo select src from shape where id=1 \; | sqlite3 -noheader g4_01.db > b.wrl
    simon:export blyth$ diff a.wrl b.wrl
    simon:export blyth$ 



"""
import os, sys, logging, string
import numpy
from env.db.simtab import Table
from md5 import md5
log = logging.getLogger(__name__) 


class WRLRegion(object):
    tmpl = r"""#---------- SOLID: %(x_name)s
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
%(x_points)s
				]
			}
			coordIndex [
%(x_faces)s
			]
			solid FALSE
		}
	}
"""
    def __init__(self, src, name=None, indx=None):
        self.src = map(string.rstrip,src)      # strip the newlines
        self.name = name
        self.indx = indx
        pass
        self.hash = md5("".join(self.src[1:])).hexdigest()   # NB first metadata comment line with volume name is skipped from hash source
        self.point = []
        pass
        mreg = self.map_regions(self.src)
        assert mreg['valid'], mreg
        pass
        self.src_head   = self.src[:mreg['points']+1]
        self.src_points = self.src[mreg['points']+1:mreg['-points']]
        self.src_tail   = self.src[mreg['-points']:]
        pass
        self.src_faces  = self.src[mreg['faces']+1:mreg['-faces']]
        pass
        self.npoints = len(self.src_points)

    def __repr__(self):
        return "# [%-6s] (%10s) :  %s " % ( self.indx, len(self.src), self.name )
    def __str__(self):
        return "\n".join( [repr(self), "".join(self.src)] ) 

    def map_regions(self, ori ):  
        """
        For interesting lines within the template find 
        the corresponding line indices within the original

        NB the exact template line including tabs is expected
        to be found within the original
        """
        log.debug("\n".join(ori))
        region, mreg = None, {}
        sreg = dict(points='point [',faces='coordIndex [',close=']')
        valid = True
        for line in self.tmpl.split("\n"):
            if line[-len(sreg['points']):] == sreg['points']:
                token, region= True, "points"
            elif line[-len(sreg['faces']):] == sreg['faces']:
                token, region= True, "faces" 
            elif line[-len(sreg['close']):] == sreg['close']:
                token, region= True, "-" + region 
            else:
                token = False
            if token:
                try:
                    idx = ori.index(line)   
                except ValueError:
                    log.debug("failed for find line \"%s\" in region\n%s" % (line, region) ) 
                    idx = None 
                    valid = False
                mreg[region] = idx 
            pass    
        mreg['valid'] = valid 
        return mreg 

    def parse_points(self):
        #log.info("parse_points %s " % self.npoints) 
        data = numpy.fromstring( "".join(map(lambda line:line[:-1],self.src_points)), dtype=numpy.float32, sep=' ')
        data.shape = (-1, 3)
        self.point = data
        assert self.npoints == len(self.point), (self.npoints, len(self.point)) 

    def parse_faces(self):
        fstr = "".join(self.src_faces)
        data = numpy.fromstring( fstr[:-1], dtype=numpy.int, sep=',')
        #log.info(">>>>%s<<<<<" % fstr[:-1]) 
        log.info(data) 
        self.face = data


class WRLParser(list):
    pfx_camera = '#---------- CAMERA'
    pfx_solid = '#---------- SOLID: '
    def __init__(self): 
        self.region = None
        self.lpfx_solid = len(self.pfx_solid)
        self.lpfx_camera = len(self.pfx_camera)
        self.buffer = []
        self.nregion = 0 

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
            if self.region == 'camera':
                pass
            else:    
                reg = WRLRegion( self.buffer, self.region, indx=len(self))  
                reg.parse_points()  
                reg.parse_faces()  
                self.append( reg )
            pass    
            self.buffer[:] = []

    def head(self, hdl, id, idlabel):
        s = "\n".join(hdl)
        if not idlabel:
            return s
        return s.replace('Shape', 'DEF S%s Shape' % id ).replace('Material','DEF M%s Material' % id )

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

    def save(self, path, opts):
        idlabel = opts.idlabel
        idoffset = opts.idoffset
        path = os.path.abspath(path)
        if os.path.exists(path):
            log.info("remove pre-existing db file %s " % path)
            os.remove(path)
        pass

        geom_t  = Table(path, "geom", idx="int",name="text", nvertex="int" )   # summary schema for fast comparison against daedb.py geom
        if opts.shape:
            shape_t = Table(path, "shape", id="int",name="text", src="blob", src_points="blob", src_faces="blob", src_head="blob",src_tail="blob", hash="text")
        if opts.points:
            point_t = Table(path, "point", id="int",idx="int",x="float",y="float",z="float")
        if opts.faces:
            face_t = Table(path, "face", id="int",idx="int",v0="int",v1="int",v2="int", v3="int" )

        log.info("gathering geometry, using idoffset %s idlabel %s " % (idoffset,idlabel) )  
        for rg in self:
            idx = rg.indx + idoffset

            geom_t.add(idx=idx,name=rg.name,nvertex=rg.npoints)

            if opts.shape:
                shape_t.add(id=idx,name=rg.name,hash=rg.hash,
                        src="\n".join(rg.src), 
                        src_faces="\n".join(rg.src_faces),    
                        src_points="\n".join(rg.src_points),
                        src_head=self.head(rg.src_head,idx,idlabel=idlabel),
                        src_tail="\n".join(rg.src_tail),
                        )

            if opts.points:
                for pid,xyz in enumerate(rg.point):
                    x,y,z = map(float,xyz)
                    point_t.add(id=pid,idx=idx,x=x,y=y,z=z)
            if opts.faces:
                for fid,vindices in enumerate(rg.face):
                    ii  = map(int,vindices)
                    face_t.add(id=fid,idx=idx,v0=ii[0],v1=ii[1],v2=ii[2],v3=ii[3])

            pass
        # writes to the DB a table at a time
        log.info("start persisting to %s " % path ) 
        geom_t.insert()   
        if opts.shape: 
            shape_t.insert()   
        if opts.points: 
            point_t.insert()   
        if opts.faces: 
            face_t.insert()   
        log.info("completed persisting to %s " % path ) 


    def extend(self, path, tn="xshape", name=True):
        dummy = Table(path)
        sammd_ = lambda _:"sum(%(_)s) as sum%(_)s, avg(%(_)s) as a%(_)s, min(%(_)s) as min%(_)s, max(%(_)s) as max%(_)s, max(%(_)s) - min(%(_)s) as d%(_)s" % locals() 
        xsql = "select sid, count(*) as npo, " + ",".join(map(sammd_, ("x","y","z")))
        if name:
            xsql += " ,name from point join shape on point.idx = shape.id group by idx "
        else:    
            xsql += " from point group by idx "
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
    op.add_option(      "--head", default=None, help="Quick run just on head lines of the input file.")
    op.add_option("-s", "--save", action="store_true", help="Save parsed geometry info into DB", default=False )
    op.add_option("-x", "--extend", action="store_true", help="Create the extents table from the pre-created DB.", default=False )
    op.add_option(      "--idoffset", type="int", default=0, help="Offset of shape indices. Default %default " )
    op.add_option(      "--noidlabel", action="store_false", dest="idlabel",  default=True, help="Add VRML DEF names to shapes and materials to allow EAI access. Default %default " )
    op.add_option( "-P","--nopoints", action="store_false", dest="points",  default=True, help="Skip Recording vertices into points table. Default %default " )
    op.add_option( "-F","--nofaces", action="store_false", dest="faces",  default=True, help="Skip Recording vertex indices into face table. Default %default " )
    op.add_option( "-S","--noshape",  action="store_false", dest="shape",  default=True, help="Skip Recording full shape in the shape table. Default %default " )
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


def main():
    opts, args = parse_args(__doc__)
    path = args[0]
    if not opts.head is None:
        dbpath = os.path.abspath(path + ".head%s.db" % opts.head )
    else:
        dbpath = os.path.abspath(path + ".db")

    wrlp = WRLParser()

    parse = True
    if parse:
        log.info("parse") 
        if not opts.head is None:
            head = opts.head
            wrlp(path=None,cmd="head -%(head)s %(path)s " % locals()) 
            # head only testing, must fine tune the head lines to avoid splitting regions 
        else:      
            wrlp(path)
        pass     
    else:    
        log.info("skip parse") 

    if opts.save:
        wrlp.save(dbpath, opts)
    else:    
        log.info("skip save") 

    if opts.extend:
        log.info("extend") 
        wrlp.extend(dbpath, "xshape")
    else:    
        log.info("skip extend") 

if __name__ == '__main__':
    main()

