#!/usr/bin/env python
"""

Wildcarded like query::

    simon:export blyth$ shapedb.py -k /dd/Geometry/PMT/lvPmt% -x 2700
    2013-09-11 19:09:25,802 env.geant4.geometry.export.shapecnf INFO     /Users/blyth/env/bin/shapedb.py -k /dd/Geometry/PMT/lvPmt% -x 2700
    2013-09-11 19:09:27,127 env.geant4.geometry.export.shapedb INFO     Operate on 2688 shapes, selected by opts.around "None" opts.like "/dd/Geometry/PMT/lvPmt%" query  
            sid        npo          ax          ay          az          dx          dy          dz 
           3200        338   -16587.30  -801452.67    -8842.50      262.00      277.00      198.00  /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum.1000 
           3201        482   -16635.24  -801396.40    -8842.50      164.80      150.00      196.29  /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode.1000 
           3202        242   -16579.28  -801462.12    -8842.50      165.30      152.00      196.28  /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiBottom.1001 
           3203         50   -16520.08  -801531.74    -8842.50      149.50      162.00       55.00  /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiDynode.1002 
           3206        338   -16184.36  -801006.80    -8842.50      285.30      249.00      198.00  /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum.1000 
           3207        482   -16245.24  -800964.77    -8842.50      137.40      172.00      196.29  /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode.1000 
           3208        242   -16174.18  -801013.88    -8842.50      144.00      172.00      196.28  /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiBottom.1001 
           3209         50   -16098.99  -801065.72    -8842.50      167.80      139.00       55.00  /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiDynode.1002 
           3212        338   -15910.57  -800471.74    -8842.50      295.70      205.00      198.00  /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum.1000 

Single name like query::

    simon:~ blyth$ shapedb.py -k '/dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum.1000'         
    2013-09-12 16:05:08,112 env.geant4.geometry.export.shapecnf INFO     /Users/blyth/env/bin/shapedb.py -k /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum.1000
    2013-09-12 16:05:08,114 env.geant4.geometry.export.shapedb INFO     opening /opt/local/Library/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/env/geant4/geometry/export/g4_01.db 
    2013-09-12 16:05:08,346 env.geant4.geometry.export.shapedb INFO     Operate on 672 shapes, selected by opts.around "None" opts.like "/dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum.1000" query  
            sid        npo          ax          ay          az          dx          dy          dz 
           3200        338   -16587.30  -801452.67    -8842.50      262.00      277.00      198.00  /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum.1000 
           3206        338   -16184.36  -801006.80    -8842.50      285.30      249.00      198.00  /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum.1000 
           3212        338   -15910.57  -800471.74    -8842.50      295.70      205.00      198.00  /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum.1000 


Comma delimited like query::

    simon:~ blyth$ shapedb.py -k '/dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum.1000,/dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode.1000'  -x 1400   
    2013-09-12 16:51:22,533 env.geant4.geometry.export.shapecnf INFO     /Users/blyth/env/bin/shapedb.py -k /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum.1000,/dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode.1000 -x 1400
    2013-09-12 16:51:22,535 env.geant4.geometry.export.shapedb INFO     opening /opt/local/Library/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/env/geant4/geometry/export/g4_01.db 
    2013-09-12 16:51:22,714 env.geant4.geometry.export.shapedb INFO     Operate on 1344 shapes, selected by opts.around "None" opts.like "/dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum.1000,/dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode.1000" query  
            sid        npo          ax          ay          az          dx          dy          dz 
           3200        338   -16587.30  -801452.67    -8842.50      262.00      277.00      198.00  /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum.1000 
           3201        482   -16635.24  -801396.40    -8842.50      164.80      150.00      196.29  /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode.1000 
           3206        338   -16184.36  -801006.80    -8842.50      285.30      249.00      198.00  /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum.1000 
           3207        482   -16245.24  -800964.77    -8842.50      137.40      172.00      196.29  /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode.1000 

::

    In [128]: 672*4
    Out[128]: 2688



Name diddling
-------------

::

    sqlite> select substr(src_head,0,instr(src_head,x'0A')+1)||'DEF S'||id||substr(src_head,instr(src_head,x'0A')+1) from shape limit 10,1 ;
    #---------- SOLID: /dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:4#pvStrip14Unit.4
    DEF S11 Shape {
                    appearance Appearance {
                            material Material {
                                    diffuseColor 1 1 1
                                    transparency 0.7
                            }
                    }
                    geometry IndexedFaceSet {
                            coord Coordinate {
                                    point [
    sqlite> 


    select replace(src_head,'Material {','DEF M'||id||' Material {') from shape limit 2 ;




"""
import os, sys, logging, math
log = logging.getLogger(__name__)
from env.db.simtab import Table
from shapecnf import parse_args



class Viewpoints(dict):
    """
    http://www.c3.hu/cryptogram/vrmltut/part2.html
    http://graphcomp.com/info/specs/sgi/vrml/spec/
    http://graphcomp.com/info/specs/sgi/vrml/spec/part1/nodesRef.html#Viewpoint

    The position and orientation fields of the Viewpoint node specify relative
    locations in the local coordinate system. Position is relative to the
    coordinate system's origin (0,0,0), while orientation specifies a rotation
    relative to the default orientation; the default orientation has the user
    looking down the -Z axis with +X to the right and +Y straight up. Viewpoints
    are affected by the transformation hierarchy.

    ::

                 +Y    
                  |
                  |
                  |
                  |
                  |
                  |
                  + ------------ +X
                 /
                /
               /
              /
          +Z /  


    http://graphcomp.com/info/specs/sgi/vrml/spec/part1/fieldsRef.html#SFRotation

    The SFRotation field and event specifies one arbitrary rotation, and the
    MFRotation field and event specifies zero or more arbitrary rotations.
    S/MFRotations are written to file as four floating point values separated by
    whitespace. The first three values specify a normalized rotation axis vector
    about which the rotation takes place. The fourth value specifies the amount of
    right-handed rotation about that axis, in radians. For example, an SFRotation
    containing a 180 degree rotation about the Y axis is::

        fooRot 0.0 1.0 0.0  3.14159265

    """
    tmpl = r"""

DEF V0 Viewpoint {
    position 0 0 %(dist)s    # position of camera is along +Z, default orientation looking back at the origin +X to the right 
    orientation 0 0 1 0      # orientation defined by rotation applied to the default, here zero rotation around Z axis 
    description "0"
}

DEF Vfront Viewpoint {
    position 0 0 %(dist)s    # position of camera is along +Z, default orientation looking back at the origin +X to the right 
    orientation 0 0 1 0      # orientation defined by rotation applied to the default, here zero rotation around Z axis 
    description "front"
}

DEF Vback Viewpoint {
    position 0 0 -%(dist)s    # position along -Z 
    orientation 0 1 0 %(pi)s  # the camera is rotated around the Y axis 180 degrees
    description "back"
}

DEF Vright Viewpoint {
    position %(dist)s 0 0      # position along +X
    orientation 0 1 0 %(hpi)s  #  the camera is rotated around the Y axis 90 degrees
    description "right"
}

DEF Vleft Viewpoint {
    position -%(dist)s 0 0      # position along -X
    orientation 0 1 0 -%(hpi)s  # the camera is rotated around the Y axis by -90 degrees
    description "left"
}

DEF Vtop Viewpoint {
    position 0 %(dist)s 0       # position along +Y
    orientation 1 0 0 -%(hpi)s  # the camera is rotated around the X axis -90 degrees
    description "top"
}

DEF Vbottom Viewpoint {
    position 0 -%(dist)s 0      #  position along -Y  
    orientation 1 0 0 %(hpi)s   # the camera is rotated around the X axis 90 degrees
    description "bottom"
}


    """
    def __init__(self, dist=10000):
        dict.__init__(self, dist=dist, pi=math.pi, hpi=math.pi/2.)
    def __str__(self):
        return self.tmpl % self 
         
    
    




class ShapeDB(Table):
    header=r"""#VRML V2.0 utf8
# Generated by VRML 2.0 driver of GEANT4

"""
    defcam=r"""
#---------- CAMERA
Viewpoint {
	position 0 0 111042
}
"""
    group_open = r"""
DEF %(group)s Group {
     children [
"""
    group_close = r"""
]
}
""" 

    anchor_open = r"""
     Anchor {
        description %(lquo)s%(description)s%(rquo)s
        url %(lquo)s%(url)s%(rquo)s
        children [
"""
    anchor_close = r"""
]
}
"""

    def __init__(self, path=None, tn=None ):
        path = os.path.abspath(path)
        log.info("opening %s " % path)
        Table.__init__(self, path, tn )

    def qids(self, sql ): 
        return map(lambda _:int(_[0]), self(sql))

    def group_points_sql(self, ids, opts):
        """
        Using `group by having` seems much slower
        """
        sids = ",".join(map(str,ids))
        cxyz, sxyz = opts.center, opts.scale 

        # special case identity transform, to allow exact diff check
        # BUT not working see number formatting differences
        if cxyz is None and sxyz is None:
            sql_point_xyz  = "x||' '||y||' '||z||','"
        else:    
            if cxyz is None:
                cx, cy, cz = 0,0,0
            else:
                cx, cy, cz = map(float,cxyz)
            if sxyz is None:
                sx, sy, sz = 1,1,1
            else:
                sx, sy, sz = map(float,sxyz)
            pass
            sql_point_xyz  = "(%(sx)s*(x-(%(cx)s)))||' '||(%(sy)s*(y-(%(cy)s)))||' '||(%(sz)s*(z-(%(cz)s)))||','"

        # ascii codes 09:TAB, 0A:LF 
        if opts.nameshape:
            assert 0, "nameshape option is no longer needed, as do this are initial vrml2file.py "
            sql_head = "substr(src_head,0,instr(src_head,x'0A')+1)||'DEF S'||shape.id||' '||substr(src_head,instr(src_head,x'0A')+1)"
        else:
            sql_head = "src_head"



        # hmm when everyshape has an anchor it is more difficult to navigate 
        sql_pre = ""
        sql_post = ""
        if opts.urlanchor:
            sql_pre = self.anchor_open % dict(lquo="\"||x'22'||",rquo="||x'22'",description="shape.id||x'3F'||shape.name",url="\"\"")
            sql_post = self.anchor_close

        def q_(lsql):
            lines = lsql.split("\n")
            qlines = []
            for line in lines:
                if line[-7:] == "||x'22'":
                    ql = "\"%s" % line
                else:
                    ql = "\"%s\"" % line
                qlines.append(ql)
            return qlines


        sql_tabs =  "x'09'||x'09'||x'09'||x'09'||x'09'||"
        sql_body = "group_concat(" + sql_tabs + sql_point_xyz + ",x'0A')"
        sql_tail = "src_tail"
        sql_from = "from point join shape on shape.id = point.sid where sid in (%(sids)s) group by sid ;"
        sql = "select " + "||x'0A'||".join(q_(sql_pre)+[sql_head,sql_body,sql_tail]+q_(sql_post)) + " " + sql_from 
        return sql % locals()

    def around_query(self, xyzd, like, fields="sid"):
        if xyzd is None:
            around_ = "1"
        else:    
            vals = map(float, xyzd.split(","))
            if len(vals) == 4:
                x,y,z,d = vals
                dx,dy,dz = d, d, d
            elif len(vals) == 6:
                x,y,z,dx,dy,dz = vals
            else:
                assert 0, "unsupported about parameters, expecting either 4 or 6 comma delimited floats "
            pass
            around_ = " abs(ax-(%(x)s)) < %(dx)s and abs(ay-(%(y)s)) < %(dy)s and abs(az-(%(z)s)) < %(dz)s " % locals()

        if like is None:
            like_ = "1"
        else:    
            like_ = "( " +" or ".join(map(lambda _:"name like '%s'" % _,like.split(","))) + " )"
        pass    
        return "select %(fields)s from xshape where %(around_)s and %(like_)s ;" % locals()

    def around(self, xyzd, like ):
        sql = self.around_query(xyzd, like )
        return self.qids(sql)

    def dump_query(self, ids, fields):
        sids = ",".join(map(str,ids))
        return "select %(fields)s,name from xshape where sid in (%(sids)s) ;" % locals()

    def dump(self, ids, xfields="ax,ay,az,dx,dy,dz", maxids=1000):
        xfields = xfields.split(",")
        fmt =  "# %10d %10d " + " %10.2f " * len(xfields) + " %s "
        lfmt = "# %10s %10s " + " %10s " * len(xfields)
        fields = ["sid","npo"] + xfields

        sql = self.dump_query(ids, ",".join(fields))
        log.info(lfmt % tuple(fields))
        if len(ids) < maxids:
            for _ in self(sql):
                log.info(fmt % _)
        else:
            log.info("too many ids to dump %s maxids %s , restrict selection and try again " % (len(ids),maxids) )

    def print_(self, ids, opts):
        """
        Doing all at once gives::

            sqlite3.OperationalError: disk I/O error

        """
        if not opts.dryrun:
            print self.header
            print Viewpoints(dist=opts.distance)
        if opts.group:
            print self.group_open % dict(group=opts.group)

        chunksize = opts.chunksize
        for x in range(0,len(ids),chunksize):
            chunk_ids = ids[x:x+chunksize] 
            self.dump(chunk_ids, maxids=opts.maxids)
            sql = self.group_points_sql(chunk_ids, opts)
            log.info(sql)
            if not opts.dryrun:
                batch = self.all(sql)
                for sh in batch:
                    print sh[0]

        if opts.group:
            print self.group_close % dict(group=opts.group)



    def centroid(self, ids):
        """
        Averaging a list of volumes
        -----------------------------

        Use a pair from the degenerate dozen to demo shapeset averaging::

            sqlite> select npo, sumx, sumy, sumz, sumx/npo, sumy/npo, sumz/npo, ax,ay,az  from xshape where sid in (6400,6401) ;
            npo         sumx        sumy         sumz        sumx/npo    sumy/npo    sumz/npo           ax          ay          az               
            ----------  ----------  -----------  ----------  ----------  ----------  -----------------  ----------  ----------  -----------------
            50          -797559.0   -40289110.0  -207856.0   -15951.18   -805782.2   -4157.12000000001  -15951.18   -805782.2   -4157.12000000001
            50          -797559.0   -40289110.0  -207856.0   -15951.18   -805782.2   -4157.12000000001  -15951.18   -805782.2   -4157.12000000001

            sqlite> select sum(sumx)/sum(npo) as ssx, sum(sumy)/sum(npo) as ssy, sum(sumz)/sum(npo) as ssz from  xshape where sid in (6400,6401) ;
            ssx         ssy         ssz              
            ----------  ----------  -----------------
            -15951.18   -805782.2   -4157.12000000001

        """
        sids = ",".join(map(str,ids))
        sql = "select sum(sumx)/sum(npo) as ssx, sum(sumy)/sum(npo) as ssy, sum(sumz)/sum(npo) as ssz from  xshape where sid in (%s) " % sids
        lret = map(lambda _:(_[0],_[1],_[2]),self(sql))
        assert len(lret) == 1 , (lret, sql)
        return lret[0]


    def handle_input(self, opts, args):
        annotate = False
        if len(args)>0:
            ids = map(int,args)
            nds = filter(lambda _:_ < 0,ids)
            pds = filter(lambda _:_ >= 0,ids)
            ids = sorted(map(abs, ids))
            annotate = len(nds) > 0  
            log.info("Operate on %s shapes [negated %s], selected by args : %s " % ( len(ids), len(nds), args) )
        else:
            if opts.around or opts.like:
                ids = self.around( opts.around, opts.like )
                log.info("Operate on %s shapes, selected by opts.around \"%s\" opts.like \"%s\" query  " % (len(ids),opts.around, opts.like) )
            elif opts.query:
                ids = self.qids(opts.query)
                log.info("Operate on %s shapes, selected by opts.query \"%s\" " % (len(ids),opts.query) )
            else:
                ids = None

        if opts.center:
            xyz = self.centroid(ids)
            log.info("opts.center selected, will translate all %s shapes such that centroid of all is at origin, original coordinate centroid at %s " % (len(ids), xyz))
            opts.center = xyz 

        if opts.scale:
            xyz = map(float,(opts.scale, opts.scale, opts.scale))
            log.info("opts.scale selected, will scale all %s shapes sxyz %s " % (len(ids), xyz))
            opts.scale = xyz 

        if annotate:
             log.info("presence of negated ids signals split print, negated ids will be annotated (useful for small volumes, not so good with big ones as interferes with navigation)")
             self.print_( map(abs,nds), opts )
             opts_urlanchor = opts.urlanchor
             opts.urlanchor = False
             self.print_( map(abs,pds), opts )
             opts.urlanchor = opts_urlanchor
        else:
             self.print_(ids, opts )
        return ids

def main():
    opts, args = parse_args(__doc__)
    db = ShapeDB(opts.dbpath)
    ids = db.handle_input( opts, args )

   
if __name__ == '__main__':
    main()




