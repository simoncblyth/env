#!/usr/bin/env python
"""
Shape VRML2 access by index
==============================

Shape random access and transformations

::

   ./shape.py --help
   ./shape.py -T   
   ./shape.py 1 
   ./shape.py --mode ori 12000

   ./shape.py --mode ori 1      # original with minimal touch
   ./shape.py --mode dup 1      # duplicated at line splitting level, should always match ori 
   ./shape.py --mode gen 1      # potentially with transformations


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

#. remove duplication between this and vrml2file.py 
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
         self.collect_points()

    def map_regions(self, ori ):  
        """
        For interesting lines within the template find 
        the corresponding line indices within the original

        NN the exact template line including tabs is expected
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

    def collect_points(self):
         log.info("collecting points")
         for xyz in self.db("select x,y,z from point where sid=%(sid)s" % self ):
             self.points.append(xyz)
         log.info("collected %s  points" % len(self.points))
    def gen_points(self):
        if self.opts.center_xyz is None:
            cx,cy,cz =  0.,0.,0.
        else:    
            cx,cy,cz = map(float, self.opts.center_xyz)

        if self.opts.scale is None:
            sx,sy,sz =  1.,1.,1.
        else:
            sc = float(self.opts.scale)
            sx,sy,sz =  sc,sc,sc

        # this should be using numpy, could do with an sqlite numpy inteface too
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

    def original_npo(self):
         return self.db.getone("select npo from xshape where sid=%(sid)s" % self )
    def original_src(self):
         return self.db.getone("select src from shape where id=%(sid)s" % self )
    def original_name(self):
         return self.db.getone("select name from shape where id=%(sid)s" % self )

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
    db = ShapeDB()

    if len(args)>0:
        ids = sorted(map(int,args))
        log.info("Operate on %s shapes, selected by args : %s " % ( len(ids), ids) )
    else:
        if opts.around:
            ids = db.around( opts.around )
            log.info("Operate on %s shapes, selected by opts.around query \"%s\"  " % (len(ids),opts.around) )
        elif opts.query:
            ids = db.qids(opts.query)
            log.info("Operate on %s shapes, selected by opts.query \"%s\" " % (len(ids),opts.query) )
        else:
            pass

    if opts.center:
        xyz = db.centroid(ids)
        log.info("opts.center selected, will translate all %s shapes such that centroid of all is at origin, original coordinate centroid at %s " % (len(ids), xyz))
        opts.center_xyz = xyz 
    else:
        opts.center_xyz = None

    if opts.dryrun:
        return

    for sid in ids: 
        shape = Shape(sid, opts=opts, db=db)
        if opts.DUMP:
            shape.dump()
        else:
            print str(shape)


if __name__ == '__main__':
    main()
    



