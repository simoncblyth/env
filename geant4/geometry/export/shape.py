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

TODO
-----

#. remove duplication between this and vrml2file.py 
#. name metadata, just like in original .WRL
#. extra metadata using xshape extents table
#. test full duplication of the original 82M .WRL
#. transformations using the xshape extents table

"""
import os, sys, logging
log = logging.getLogger(__name__)
from env.db.simtab import Table
from random import randrange


class Shape(dict):
    """
    NB the tabs in the template must be retained in order 
    for original mapping to work
    """
    tmpl = r"""	Shape {
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
    default_path = os.path.join(os.path.dirname(__file__),"g4_00.db")

    @classmethod
    def dbase(cls, path=None):
        if path is None:
            path = cls.default_path
        return Table(os.path.abspath(path), None)

    def __init__(self, sid, mode="gen", db=None, path=None ):
         if db is None:
             db = self.dbase(path) 
         self.db = db 
         self['sid'] = sid
         self['mode'] = mode
         self['npo'] = int(self.original_npo())
         ori = self.original_src().split("\n")
         mreg = self.map_regions(ori)

         self['ori'] = "\n".join(ori)

         # line by line reconstruct from original
         self['ori_points'] = "\n".join(ori[mreg['points']+1:mreg['-points']])
         self['ori_faces'] = "\n".join(ori[mreg['faces']+1:mreg['-faces']])

         self.mreg = mreg
         log.info("sid %(sid)s npo %(npo)s mode %(mode)s " % self )
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
         def fmt_point(xyz):
             return "\t\t\t\t\t%s %s %s," % xyz
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
         if mode == "dup":
             ctx['x_points'] = self["ori_points"]
             ctx['x_faces'] = self["ori_faces"]
         elif mode == "gen":
             ctx['x_points'] = self.gen_points()
             ctx['x_faces'] = self["ori_faces"]
         else:
             ctx['x_points'] = "# mode %(mode)s not implemented "  % locals()
             ctx['x_faces'] = "# mode %(mode)s not implemented " % locals()
         return self.tmpl % ctx

    def original_npo(self):
         return self.db.getone("select npo from xshape where sid=%(sid)s" % self )
    def original_src(self):
         return self.db.getone("select src from shape where id=%(sid)s" % self )

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
    for n in range(100):
        sid = randrange(1, nid+1)
        yield check_shape, sid, db 


def parse_args(doc):
    """
    """
    from optparse import OptionParser
    op = OptionParser(usage=doc)
    op.add_option("-o", "--logpath", default=None )
    op.add_option("-l", "--loglevel",   default="INFO", help="logging level : INFO, WARN, DEBUG ... Default %default"  )
    op.add_option("-f", "--logformat", default="%(asctime)s %(name)s %(levelname)-8s %(message)s" )
    op.add_option("-m", "--mode", default="gen", help="Mode of output: ori, dup, gen " )
    op.add_option("-T", "--TEST", action="store_true", help="Duplication testing  multiple randomly chosen shapes." )
    op.add_option("-D", "--DUMP", action="store_true", help="Debug dumping." )
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
    db = Shape.dbase()
    if opts.TEST:
        for _ in test_shape():
            _[0](*_[1:])
    elif opts.DUMP:
        sid = args[0]
        shape = Shape(sid, mode=opts.mode, db=db)
        shape.dump()
    else:
        sid = args[0]
        shape = Shape(sid, mode=opts.mode, db=db)
        print str(shape)


if __name__ == '__main__':
    main()
    



