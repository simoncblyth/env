#!/usr/bin/env python
"""
DAEGEOM
========

Dump geometry information addressed by PV index, and compare with 
info gleaned from VRML2 exports.

Usage with default dae file::

    simon:~ blyth$ daegeom.py 1000
    2013-10-31 20:10:47,914 env.graphics.collada.pycollada.daegeom INFO     /Users/blyth/env/bin/daegeom.py 1000
    2013-10-31 20:10:47,917 env.graphics.collada.pycollada.daegeom INFO     loading /usr/local/env/geant4/geometry/xdae/g4_01.dae 
    2013-10-31 20:10:56,084 env.graphics.collada.pycollada.daegeom INFO     dump_geom from /usr/local/env/geant4/geometry/xdae/g4_01.dae boundgeom index 1000 
    bpl <BoundPolylist length=6> nvtx: 8
    [[ -19283.215125   -800360.74824449   -1364.19516943]
     [ -20404.93646086 -798609.13837592   -1364.19516943]
     [ -20623.6247397  -798749.18514814   -1376.9351691 ]
     [ -19501.90340384 -800500.79501671   -1376.9351691 ]
     [ -19283.29765277 -800360.80109482   -1362.19757138]
     [ -20405.01898863 -798609.19122625   -1362.19757138]
     [ -20623.70726747 -798749.23799847   -1374.93757105]
     [ -19501.98593161 -800500.84786704   -1374.93757105]]
    from VRML2DB: 8 
    [[ -19283.19921875 -800361.           -1364.18994141]
     [ -20404.90039062 -798609.           -1364.18994141]
     [ -20623.59960938 -798749.           -1376.93005371]
     [ -19501.90039062 -800501.           -1376.93005371]
     [ -19283.30078125 -800361.           -1362.18994141]
     [ -20405.         -798609.           -1362.18994141]
     [ -20623.69921875 -798749.           -1374.93005371]
     [ -19502.         -800501.           -1374.93005371]]

When using subgeometries the coordinates will not align, even when using index offset
as different node transformations are applied to get to the targeted volume::

    daegeom.py -p $LOCAL_BASE/env/graphics/collada/3199.dae 0 -x 3199

"""
import os, logging, sys

import numpy 
import collada

# FOLLOWING DOING THIS IN G4DAEWrite ITS WRONG TO DOUBLE FIX
#from monkey_matrix_load import _monkey_matrix_load
#collada.scene.MatrixTransform.load = staticmethod(_monkey_matrix_load)

from env.geant4.geometry.vrml2.vrml2db import VRML2DB


log = logging.getLogger(__name__)

def primfix(self):
    """
    **RENDERED OBSOLETE BY _MONKEY_MATRIX_LOAD**

    Original pycollada lineset,polylist,triangleset
    use post multiplication by the transposed rotation portion of the node matrix and 
    broadcast addition of the translation portion::

        M = numpy.asmatrix(matrix).transpose()
        self._vertex = None if pl._vertex is None else numpy.asarray(pl._vertex * M[:3,:3]) + matrix[:3,3]
        self._normal = None if pl._normal is None else numpy.asarray(pl._normal * M[:3,:3])

    Easier to stay using post-multiplication to be consistent with the recursive transformations, 
    but not to transpose the rotation matrix (same as invert for a rotation matrix)
    like pycollada does (or store the transposed to avoid 

    Initial fix used::

        M = numpy.asmatrix(self.matrix).transpose()
        self._vertex = numpy.asarray(( M[:3,:3] * self.original._vertex.T ).T ) + self.matrix[:3,3]
        self._normal = numpy.asarray(( M[:3,:3] * self.original._normal.T ).T ) 

    But can avoid all the transposing by post-multiplying the untransposed original matrix. Using 
    numpy.dot avoids conversion from numpy array to matrix.

    To avoid changing pycollada here and in the recursive transformations
    could change the collada file to store the transposed rotation 
    and the translation as is.

    This works for PV1, but not below.
    """
    assert self.__class__.__name__ in ('BoundLineSet','BoundPolylist','BoundTriangleSet'), self
    if self.original._vertex is None:
        self._vertex = None 
    else: 
        self._vertex = numpy.dot(self.original._vertex,self.matrix[:3,:3]) + self.matrix[:3,3]

    if self.original._normal is None:
        self._normal = None 
    else: 
        self._normal = numpy.dot(self.original._normal, self.matrix[:3,:3]) 


def dump_geom(args, opts):

    rindex = int(args[0])    # root relative index
    aindex = int(args[0]) + opts.indexoffset   # absolute index

    log.info("root relative index %s absolute index %s " % ( rindex, aindex )) 

    path = opts.daepath
    log.info("loading %s " % path )
    dae = collada.Collada(path)

    top = dae.scene.nodes[0]
    log.info("dump_geom from %s boundgeom rindex %s " % (path, rindex))

    boundgeom = list(top.objects('geometry'))
    bg = boundgeom[rindex]

    prim = list(bg.primitives())
    assert len(prim) == 1, len(prim)
    bp = prim[0]
    bpl = list(bg.primitives())[0] 

    print "bpl", bpl, "nvtx:", len(bpl.vertex)
    print bpl.vertex

    if opts.primitives:
        for i, po in enumerate(bpl):
            print i, po, po.indices

    #  primfix not needed when using monkey patched matrix loading 
    #primfix(bpl)
    #print "after primfix", bpl, "nvtx:", len(bpl.vertex)
    #print bpl.vertex

    db = VRML2DB()
    sh = db.shape(aindex)
    vpo = sh.points
    print "from VRML2DB: aindex %s %s \n" % (aindex, len(vpo)) , vpo
    for _ in sh.faces:print _

class Defaults(object):
    logformat = "%(asctime)s %(name)s %(levelname)-8s %(message)s"
    loglevel = "INFO"
    logpath = None
    primitives = False
    daepath = "$LOCAL_BASE/env/geant4/geometry/xdae/g4_01.dae"
    indexoffset = 0

def parse_args(doc):
    from optparse import OptionParser
    defopts = Defaults()
    op = OptionParser(usage=doc)

    op.add_option("-o", "--logpath", default=defopts.logpath )
    op.add_option("-l", "--loglevel",   default=defopts.loglevel, help="logging level : INFO, WARN, DEBUG ... Default %default"  )
    op.add_option("-f", "--logformat", default=defopts.logformat )
    op.add_option("-m", "--primitives", action="store_true",  default=defopts.primitives )
    op.add_option("-p", "--daepath", default=defopts.daepath )
    op.add_option("-x", "--indexoffset", type=int, default=defopts.indexoffset )

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
        logging.basicConfig(format=opts.logformat,level=level)
    pass
    log.info(" ".join(sys.argv))

    daepath = os.path.expandvars(os.path.expanduser(opts.daepath))
    if not daepath[0] == '/':
        daepath = os.path.abspath(daepath)
    assert os.path.exists(daepath), (daepath,"DAE file not at the new expected location, please create the directory and move the .dae  there, please")
    opts.daepath = daepath
    pass 
    return opts, args


def main():
    opts, args = parse_args(__doc__)
    dump_geom(args, opts)


if __name__ == '__main__':
    main()


