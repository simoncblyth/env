#!/usr/bin/env python
"""

"""
import os, logging, argparse
import numpy as np
log = logging.getLogger(__name__)

def parse_args(doc):
    defaults = {}
    defaults['nodes']="3153:12230"
    #defaults['nodes']="5000:5100"   # some PMTs for quick testing

    defaults['size']="1440,852"
    #defaults['size']="640,480"

    defaults['path'] = os.environ['DAE_NAME']

    parser = argparse.ArgumentParser(doc)
    parser.add_argument("-n","--nodes", default=defaults['nodes'],   help="DAENode.getall node(s) specifier %(default)s",type=str)
    parser.add_argument(     "--size", default=defaults['size'], help="Pixel size  %(default)s", type=str)
    parser.add_argument(     "--path", default=defaults['path'], help="Path of geometry file  %(default)s",type=str)

    parser.add_argument(  "--line", dest="line", action="store_true", help="Switch on line mode polygons  %(default)s" )

    parser.add_argument("--nolight",dest="light", action="store_false", help="Inhibit light setup  %(default)s" )
    parser.add_argument("--nofill", dest="fill", action="store_false", help="Inhibit fill mode polygons  %(default)s" )
    parser.add_argument("--notransparent",dest="transparent", action="store_false", help="Inhibit transparent fill  %(default)s" )

    parser.add_argument("--rgba",  default=".7,.7,.7,.5", help="RGBA color of geometry, the alpha has a dramatic effect  %(default)s",type=str)
    parser.add_argument("--frame",  default="1,1", help="Viewport framing  %(default)s",type=str)

    parser.add_argument("-l","--loglevel", default="INFO", help="INFO/DEBUG/WARN/..   %(default)s")  
    
    parser.add_argument(     "--yfov",  default=50., help="Initial vertical field of view in degrees. %(default)s", type=float)
    parser.add_argument(     "--near",  default=0.001, help="Initial near. %(default)s", type=float)
    parser.add_argument(     "--far",  default=100., help="Initial far. %(default)s", type=float)
    parser.add_argument(     "--thetaphi",  default="0,0", help="Initial theta,phi. %(default)s", type=str)
    parser.add_argument(     "--xyz",  default="0,0,3", help="Initial viewpoint in canonical -1:1 cube coordinates %(default)s", type=str)
    parser.add_argument(     "--parallel", action="store_true", help="Parallel projection, aka orthographic." )
    parser.add_argument(     "--fullscreen", action="store_true", help="Start in fullscreen mode." )

    # target based positioning mode switched on by presence of target 
    parser.add_argument("-t","--target", default=None,     help="Node specification of solid on which to focus or empty string for all",type=str)
    parser.add_argument("-e","--eye",   default="-2,0,0", help="Eye position",type=str)
    parser.add_argument("-a","--look",  default="0,0,0",   help="Lookat position",type=str)
    parser.add_argument("-u","--up",   default="0,0,1", help="Eye position",type=str)


    # not yet implemented
    parser.add_argument("-F","--noflip",  dest="flip", action="store_false", default=True, help="Pixel y flip.")
    parser.add_argument("-s","--pscale", default=1., help="Parallel projection, scale.", type=float  )
    parser.add_argument("-i","--interactive", action="store_true", help="Interative Mode")
    parser.add_argument("-j","--jump", default=None, help="Animated transition to another node.")  

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.loglevel))
    
    fvec_ = lambda _:map(float, _.split(","))
    ivec_ = lambda _:map(int, _.split(","))

    args.thetaphi = fvec_(args.thetaphi) 
    args.xyz = fvec_(args.xyz) 

    args.frame = fvec_(args.frame) 
    args.rgba = fvec_(args.rgba) 
    args.eye = fvec_(args.eye) 
    args.look = fvec_(args.look) 
    args.up = fvec_(args.up) 
    args.size = ivec_(args.size) 


    if args.target is None:
       if args.near < 1.:
           args.near = 1. 
           log.warn("amending near to %s for non-target mode" % args.near)

    return args



class DAEConfig(object):
    def __init__(self, doc):
        self.args = parse_args(doc) 
        np.set_printoptions(precision=4, suppress=True)


if __name__ == '__main__':
    cfg = DAEConfig(__doc__)
    
