#!/usr/bin/env python
"""
Shape VRML2 access by index
==============================

.. warning:: DEPRECATED : use shapedb.py FOR A FASTER APPROACH, THIS WILL BE DELETED AFTER A SALVAGE HAS BEEN DONE


Shape random access and transformations

::

   ./shape.py --help
   ./shape.py -T   
   ./shape.py 1 
   ./shape.py --mode ori 12000

   ./shape.py --mode ori 1      # original with minimal touch
   ./shape.py --mode dup 1      # duplicated at line splitting level, should always match ori 
   ./shape.py --mode gen 1      # potentially with transformations



Region restriction by around query
------------------------------------

::

    simon:export blyth$ shape.py --around=-18644.9,-798825.5,-8993.5,1000,1000,4000 --center > cc.wrl
    2013-09-06 19:52:01,291 env.geant4.geometry.export.shapecnf INFO     /Users/blyth/env/bin/shape.py --around=-18644.9,-798825.5,-8993.5,1000,1000,4000 --center
    2013-09-06 19:52:01,527 env.geant4.geometry.export.shapedb INFO     Operate on 100 shapes, selected by opts.around query "-18644.9,-798825.5,-8993.5,1000,1000,4000"  
    2013-09-06 19:52:01,726 env.geant4.geometry.export.shapedb INFO     opts.center selected, will translate all 100 shapes such that centroid of all is at origin, original coordinate centroid at (-19222.868650817793, -798345.191159482, -7550.7662023469311) 

Region restriction by like query on name
------------------------------------------

::

    simon:export blyth$ shape.py -k /dd/Geometry/PMT/lvPmt% > pmts.wrl 
    2013-09-11 19:12:01,117 env.geant4.geometry.export.shapecnf INFO     /Users/blyth/env/bin/shape.py -k /dd/Geometry/PMT/lvPmt%
    2013-09-11 19:12:02,457 env.geant4.geometry.export.shapedb INFO     Operate on 2688 shapes, selected by opts.around "None" opts.like "/dd/Geometry/PMT/lvPmt%" query  
    2013-09-11 19:12:04,539 env.geant4.geometry.export.shape INFO     {'-faces': 856, '-points': 349, 'points': 10, 'faces': 351}
    2013-09-11 19:12:04,556 env.geant4.geometry.export.shape INFO     sid 3200 npo 338 mode gen name /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum.1000
    2013-09-11 19:12:04,557 env.geant4.geometry.export.shape INFO     collecting points faster
    2013-09-11 19:12:06,055 env.geant4.geometry.export.shape INFO     collected 338  points faster
    ...


.. warning:: Painfully slow for large numbers of shapes


Region restriction by query
----------------------------

::

    simon:export blyth$ ./shape.py --query "select sid from xshape where abs(ax+16444.75) < 100 and abs(ay+811537.5) < 100 ;" --center > ss.wrl 
    2013-09-04 20:36:03,791 __main__ INFO     ./shape.py --query select sid from xshape where abs(ax+16444.75) < 100 and abs(ay+811537.5) < 100 ; --center
    2013-09-04 20:36:03,928 __main__ INFO     getting 12 ids from opts.query "select sid from xshape where abs(ax+16444.75) < 100 and abs(ay+811537.5) < 100 ;" 
    2013-09-04 20:36:04,016 __main__ INFO     ShapeSet center of 12 shapes is at (-16504.801041666666, -811595.20833333337, -1337.5170833333334) 
    2013-09-04 20:36:06,313 __main__ INFO     {'-faces': 28, '-points': 19, 'points': 10, 'faces': 21}
    2013-09-04 20:36:06,315 __main__ INFO     sid 299 npo 8 mode gen name /dd/Geometry/RPC/lvNearRPCRoof#pvNearUnSlopModArray#pvNearUnSlopModOne:1#pvNearUnSlopMod:6#pvNearSlopModUnit.1
    2013-09-04 20:36:06,317 __main__ INFO     collecting points
    2013-09-04 20:36:07,977 __main__ INFO     collected 8  points
    2013-09-04 20:36:10,310 __main__ INFO     {'-faces': 28, '-points': 19, 'points': 10, 'faces': 21}
    2013-09-04 20:36:10,312 __main__ INFO     sid 300 npo 8 mode gen name /dd/Geometry/RPC/lvRPCMod#pvRPCFoam.1000
    2013-09-04 20:36:10,313 __main__ INFO     collecting points
    ... 


TODO
-----

#. move slow manipulations (and duplications) upstream to vrml2file.py 
#. extra metadata using xshape extents table
#. test full duplication of the original 82M .WRL


"""
import os, sys, logging
log = logging.getLogger(__name__)
from random import randrange
from shapecnf import parse_args
from shapedb import ShapeDB

class Shape(dict):
    """
    NB the tabs in the template must be retained in order 
    for original mapping to work
    """
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

    def __init__(self, sid, opts=None, db=None, path=None):
         """
         :param sid: integer shape id
         :param opts:
         :param db:
         :param path:
         """ 
         if db is None:
             db = ShapeDB(path)
         self.db = db 
         self.opts = opts
         self['sid'] = sid
         self['mode'] = opts.mode
         self['npo'] = int(self.original_npo())
         self['name'] = self.original_name()

         ori = self.original_src().split("\n")
         mreg = self.map_regions(ori)

         # line by line reconstruct from original
         self['ori'] = "\n".join(ori)
         self['ori_points'] = "\n".join(ori[mreg['points']+1:mreg['-points']])
         self['ori_faces'] = "\n".join(ori[mreg['faces']+1:mreg['-faces']])

         self.mreg = mreg
         log.info("sid %(sid)s npo %(npo)s mode %(mode)s name %(name)s" % self )
         self.points = []
         self.collect_points_faster()

    def original_load(self):
        sql = "select name, src, npo from shape join xshape on shape.id = xshape.sid where shape.id=%(sid)s ;" % self  
        name,src,npo = self.db.getone(sql)
        self['name'] = name
        self['ori'] = src
        self['npo'] = npo

    def original_npo(self):
        return self.db.getone("select npo from xshape where sid=%(sid)s" % self )
    def original_src(self):
        return self.db.getone("select src from shape where id=%(sid)s" % self )
    def original_name(self):
        return self.db.getone("select name from shape where id=%(sid)s" % self )


    def map_regions(self, ori ):  
        """
        For interesting lines within the template find 
        the corresponding line indices within the original

        NB the exact template line including tabs is expected
        to be found within the original
        """
        region, mreg = None, {}
        sreg = dict(points='point [',faces='coordIndex [',close=']')
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
                mreg[region] = ori.index(line)   
            pass    
        log.info(mreg) 
        for i, line in enumerate(ori):
             log.debug("%-10s %-20s %s " % ( i, mreg.get(i,"") , line ))
        return mreg 

    def collect_points_faster(self):
        """
        Only a little faster. 
        A grouped approach would be better when pulling multiple shapes at once::

             select group_concat('\t\t\t\t'||%(sx)s*(x-(%(cx)s))||' '||%(sy)s*(y-(%(cy)s))||' '||%(sz)s*(z-(%(cz)s))||',', '\n') from point group by sid having sid=%(sid)s ;

        OR could dispense with the points table and interpret the original points.  At level of
        single shape would probably be faster. Probably not for a large number of shapes.
        """
        log.info("collecting points faster")
        self.points = self.db.all("select x,y,z from point where sid=%(sid)s" % self )
        log.info("collected %s  points faster" % len(self.points))

    def collect_points(self):
        """
        This is slow.  
         
        Better to find an sqlite numpy interface that can do a single query grab into
        an numpy array.

        http://stackoverflow.com/questions/7901853/numpy-arrays-with-sqlite
        http://code.google.com/p/esutil/source/browse/trunk/esutil/sqlite_util.py

        Need a pysqlite_numpy pysqlite fork that does type conversions from sqlite 
        into numpy dtype, just like mysql_numpy does for mysql_python 
        (actually that was a bad name mysql_python_numpy would be better)

        OR plump for pytables (based on HDF + numpy) 

        http://www.pytables.org 
        http://www.hdfgroup.org/HDF5/whatishdf5.html

        BUT, that adds a dependency/inconvenience

        """
        log.info("collecting points")
        for xyz in self.db("select x,y,z from point where sid=%(sid)s" % self ):
            self.points.append(xyz)
        log.info("collected %s  points" % len(self.points))

    def gen_points(self):
        """
        TODO: Adopt numpy for the manipulations when available

        Alternately could do the math and string formation with sqlite to keep dependencies simple
        at expense of query complexity 
        """
        if self.opts.center is None:
            cx,cy,cz =  0.,0.,0.
        else:    
            cx,cy,cz = map(float, self.opts.center)

        if self.opts.scale is None:
            sx,sy,sz =  1.,1.,1.
        else:
            sc = float(self.opts.scale)
            sx,sy,sz =  sc,sc,sc

        def fmt_point(xyz):
            txyz = (xyz[0] - cx)*sx, (xyz[1] - cy)*sy, (xyz[2] - cz)*sz
            return "\t\t\t\t\t%s %s %s," % txyz
        pass  
        return "\n".join(map(fmt_point,self.points))

    def __str__(self):
         return self.filltmpl(self['mode'])
         
    def filltmpl(self, mode):     
         """
         :param mode: ori/dup/gen 

         Meanings of the modes:

         `ori`
                return the untouched original string, for comparison against the `dupe`
         `dup`
                attempts to reconstruct the original precisely, using template with the original component
         `gen` 
                generate, potentially with translations and scalings applied to points depending on options settings

         """

         if mode == "ori":
             return self['ori'] 

         ctx = {}
         ctx['x_name'] = self["name"]
         ctx['x_faces'] = self["ori_faces"]
         if mode == "dup":
             ctx['x_points'] = self["ori_points"]
         elif mode == "gen":
             ctx['x_points'] = self.gen_points()
         else:
             ctx['x_points'] = "# mode %(mode)s not implemented "  % locals()
         return self.tmpl % ctx

    def dump(self):
        print "\n".join(["### ori", self['ori'], "#########"])
        print "\n".join(["### ori_points", self['ori_points'], "#########"])
        print "\n".join(["### ori_faces", self['ori_faces'], "#########"])
 

def check_shape(sid, db ):
    log.info("check_shape %s " % sid )
    shape = Shape(sid, mode="dup", db=db )
    ori = shape['ori']
    dup = str(shape)
    assert ori == dup , "\n".join(["####### original src", ori,"###### duped ", dup ])

def test_shape():
    db = Shape.dbase()
    nid = db.getone("select max(id) from shape")
    log.info("test_shape nid %s " % nid )
    for n in range(100):
        sid = randrange(1, nid+1)
        yield check_shape, sid, db 


def main():
    opts, args = parse_args(__doc__)
    db = ShapeDB(opts.dbpath)
    ids = db.handle_input( opts, args )

    if opts.dryrun:
        return

    for sid in ids: 
        shape = Shape(sid, opts=opts, db=db)
        if opts.DUMP:
            shape.dump()
        else:
            print str(shape)


if __name__ == '__main__':
    #main()
    print "you should be using shapedb.py not this"
    



