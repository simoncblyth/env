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

    defaults['rlight'] = "-1,1,1"
    defaults['glight'] = "1,1,1"
    defaults['blight'] = "0,-1,1"
    defaults['flight'] = 1.
    defaults['wlight'] = 1.
    defaults['lights'] = "rgb"
    defaults['near'] = 0.0001
    defaults['far'] = 100.
    defaults['yfov'] = 50.

    defaults['target'] = None
    defaults['eye'] = "-2,-2,0"
    defaults['look'] = "0,0,0"
    defaults['up'] = "0,0,1"

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
    
    parser.add_argument(     "--yfov",  default=defaults['yfov'], help="Initial vertical field of view in degrees. %(default)s", type=float)
    parser.add_argument(     "--near",  default=defaults['near'], help="Initial near in units of target extent. %(default)s", type=float)
    parser.add_argument(     "--far",  default=defaults['far'], help="Initial far in units of target extent. %(default)s", type=float)

    parser.add_argument(     "--nearclip",  default="0.0001,1000", help="Allowed range for near. %(default)s", type=str )
    parser.add_argument(     "--farclip",  default="1,100000", help="Allowed range for far. %(default)s", type=str )
    parser.add_argument(     "--thetaphi",  default="0,0", help="Initial theta,phi. %(default)s", type=str)
    parser.add_argument(     "--xyz",  default="0,0,3", help="Initial viewpoint in canonical -1:1 cube coordinates %(default)s", type=str)

    parser.add_argument(     "--parallel", action="store_true", help="Parallel projection, aka orthographic." )
    parser.add_argument(     "--fullscreen", action="store_true", help="Start in fullscreen mode." )

    # target based positioning mode switched on by presence of target 
    parser.add_argument("-t","--target", default=defaults['target'],     help="Node specification of solid on which to focus or empty string for all",type=str)
    parser.add_argument("-e","--eye",   default=defaults['eye'], help="Eye position",type=str)
    parser.add_argument("-a","--look",  default=defaults['look'],   help="Lookat position",type=str)
    parser.add_argument("-u","--up",   default=defaults['up'], help="Up direction",type=str)

    parser.add_argument(     "--rlight",  default=defaults['rlight'], help="Red light position",type=str)
    parser.add_argument(     "--glight",  default=defaults['glight'], help="Green light position",type=str)
    parser.add_argument(     "--blight",  default=defaults['blight'], help="Blue light position",type=str)
    parser.add_argument(     "--flight",  default=defaults['flight'], help="Light position scale factor",type=float)
    parser.add_argument(     "--wlight",  default=defaults['wlight'], help="Homogeonous 4th coordinate, 0 for infinity",type=float)
    parser.add_argument(     "--lights",  default=defaults['lights'], help="Enable rgb lights",type=str)
    

    parser.add_argument("-j","--jump", default=None, help="Animated transition to another node.")  
    parser.add_argument(     "--speed", default=1e-3, help="Animation interpolatiom speed, %(default)s", type=float)  

    # not yet implemented
    parser.add_argument("-F","--noflip",  dest="flip", action="store_false", default=True, help="Pixel y flip.")
    parser.add_argument("-s","--pscale", default=1., help="Parallel projection, scale.", type=float  )
    parser.add_argument("-i","--interactive", action="store_true", help="Interative Mode")

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.loglevel))
    
    fvec_ = lambda _:map(float, _.split(","))
    ivec_ = lambda _:map(int, _.split(","))

    args.thetaphi = fvec_(args.thetaphi) 
    args.xyz = fvec_(args.xyz) 

    args.nearclip = fvec_(args.nearclip) 
    args.farclip = fvec_(args.farclip) 

    args.rlight = fvec_(args.rlight) 
    args.glight = fvec_(args.glight) 
    args.blight = fvec_(args.blight) 

    args.frame = fvec_(args.frame) 
    args.rgba = fvec_(args.rgba) 
    args.eye = fvec_(args.eye) 
    args.look = fvec_(args.look) 
    args.up = fvec_(args.up) 
    args.size = ivec_(args.size) 

    return args



class DAEConfig(object):
    def __init__(self, doc):
        self.args = parse_args(doc) 
        np.set_printoptions(precision=4, suppress=True)

    def commandline(self):
        args = self.args
        return "--nodes %s --near %s --far %s --yfov %s --target %s --eye %s --look %s --up %s" % (args.nodes, args.near, args.far, args.yfov, args.target, args.eye, args.look, args.up )



if __name__ == '__main__':
    cfg = DAEConfig(__doc__)
    
