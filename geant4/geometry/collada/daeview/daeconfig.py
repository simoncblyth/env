#!/usr/bin/env python
"""
DAEConfig 
============

Argument parsing is done at command line launch as normal, and in addition
during live running a subset of options can be sent as UDP messages 
and parsed with the so called live parser.

Beware of short options and minus signs:

    -e -5,0,5         # invalid
    --eye=-5,0,5      # OK


Options that can be acted upon live are marked with an "[I]" 
in the help.  Usage::

   udp.py -t +4000


TODO: live parsing of negative toggles is non-intuitive, maybe add reverse ones too


"""
import os, sys, logging, math, socket
import argparse

try: 
    from collections import OrderedDict
except ImportError:
    OrderedDict = dict

import numpy as np

log = logging.getLogger(__name__)


def address():
    """
    Not a general solution, but working for me 

    http://stackoverflow.com/questions/166506/finding-local-ip-addresses-using-pythons-stdlib
    """
    return socket.gethostbyname(socket.gethostname())


class ArgumentParserError(Exception): pass
class ThrowingArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ArgumentParserError(message)

ivec_ = lambda _:map(int,_.split(","))
fvec_ = lambda _:map(float,_.split(","))

class DAEConfig(object):

    size = property(lambda self:ivec_(self.args.size))
    block=property(lambda self:ivec_(self.args.block))
    launch=property(lambda self:ivec_(self.args.launch))
    flags=property(lambda self:ivec_(self.args.flags))

    frame = property(lambda self:fvec_(self.args.frame))
    rgba = property(lambda self:fvec_(self.args.rgba))
    nearclip = property(lambda self:fvec_(self.args.nearclip))
    farclip = property(lambda self:fvec_(self.args.farclip))
    yfovclip = property(lambda self:fvec_(self.args.yfovclip))
    thetaphi = property(lambda self:fvec_(self.args.thetaphi))


    def __init__(self, doc):
      
        base_parser, base_defaults = self._make_base_parser(doc)
        init_parser, live_defaults = self._make_live_parser(parents=[base_parser]) 

        self.init_parser = init_parser

        defaults = OrderedDict()
        defaults.update(base_defaults)
        defaults.update(live_defaults)

        self.base_defaults = base_defaults
        self.live_defaults = live_defaults
        self.defaults = defaults

        live_parser, dummy         = self._make_live_parser(argument_default=argparse.SUPPRESS, parents=[], with_defaults=False) 
        self.live_parser = live_parser
        self.args = None

    def init_parse(self):
        try:
            args = self.init_parser.parse_args()
        except ArgumentParserError, e:
            print "ArgumentParserError %s %s " % (e, repr(sys.argv)) 
            return
        
        logging.basicConfig(level=getattr(logging, args.loglevel), format=args.logformat )
        np.set_printoptions(precision=4, suppress=True)
        self.args = args

    def live_parse(self, cmdline):
        live_args = None           
        try:
            live_args = self.live_parser.parse_args(cmdline.lstrip().rstrip().split(" "))
        except ArgumentParserError, e:
            log.info("ArgumentParserError %s while parsing %s " % (e, cmdline)) 
        pass
        return live_args

    def __call__(self, cmdline):
        return self.live_parse(cmdline)

    def _settings(self, args, defaults, all=False):
        if args is None:return "PARSE ERROR"
        if all:
            filter_ = lambda kv:True
        else:
            filter_ = lambda kv:kv[1] != getattr(args,kv[0]) 
        pass
        wid = 20
        fmt = " %-30s : %20s : %s %20s %s "
        mkr_ = lambda k:"**" if getattr(args,k) != defaults.get(k) else "  "
        return "\n".join([ fmt % (k,str(v)[:wid],mkr_(k),str(getattr(args,k))[:wid],mkr_(k)) for k,v in filter(filter_,defaults.items()) ])

    def base_settings(self, all_=False):
        return self._settings( self.args, self.base_defaults, all_ )

    def live_settings(self, all_=False):
        return self._settings( self.args, self.live_defaults, all_ )


    def report(self):
        changed = self.changed_settings()
        if len(changed.split("\n")) > 1:
            print "changed settings\n", changed
        #print "all settings\n",self.all_settings()

    def all_settings(self):
        return "\n".join(filter(None,[
                      self.base_settings(True) ,
                      "---", 
                      self.live_settings(True) 
                         ]))
    def changed_settings(self):
        return "\n".join(filter(None,[
                      self.base_settings(False) ,
                      "---", 
                      self.live_settings(False) 
                         ]))

    def __repr__(self):
        return self.changed_settings() 
    def commandline(self):
        args = self.args
        return "--nodes %s --near %s --far %s --yfov %s --target %s --eye %s --look %s --up %s" % (args.nodes, args.near, args.far, args.yfov, args.target, args.eye, args.look, args.up )

    def _make_base_parser(self, doc):
        """
        Base parser handles arguments/options that 
        must be set at initialisation, either because they 
        only make sense to be done there or due to 
        handling of live updates not being implemented.
        """
        parser = argparse.ArgumentParser(doc, add_help=False)

        defaults = OrderedDict()

        defaults['loglevel'] = "INFO"
        defaults['logformat'] = "%(asctime)-15s %(name)-20s:%(lineno)-3d %(message)s"
        defaults['host'] = os.environ.get("DAEVIEW_UDP_HOST","127.0.0.1")
        defaults['port'] = os.environ.get("DAEVIEW_UDP_PORT", "15006")
        defaults['address'] = address()
        defaults['bookmarks'] = "bookmarks.cfg"

        parser.add_argument( "--loglevel",help="INFO/DEBUG/WARN/..   %(default)s")  
        parser.add_argument( "--logformat", help="%(default)s")  
        parser.add_argument( "--host", help="Hostname to bind to for UDP messages ", type=str  )
        parser.add_argument( "--port", help="Port to bind to for UDP messages ", type=str  )
        parser.add_argument( "--address", help="IP address %(default)s", type=str  )
        parser.add_argument( "--bookmarks", help="Path to persisted bookmarks  %(default)s", type=str  )

        defaults['deviceid'] = None
        defaults['cuda_profile'] = False
        parser.add_argument(      "--device-id", help="CUDA device id.", type=str )
        parser.add_argument(      "--cuda-profile", help="Sets CUDA_PROFILE envvar.", action="store_true" )

        defaults['with_cuda_image_processor'] = False
        defaults['cuda_image_processor'] = "Invert"
        parser.add_argument( "-I","--with-cuda-image-processor", help="Enable CUDA image processors ", action="store_true"  )
        parser.add_argument(      "--cuda-image-processor", help="Name of the CUDA image processor to use.", type=str )

        defaults['with_chroma'] = False
        defaults['max_alpha_depth'] = 10
        parser.add_argument( "-C","--with-chroma", dest="with_chroma", help="Indicate if Chroma is available.", action="store_true" )
        parser.add_argument(      "--max-alpha-depth", help="Chroma Raycaster max_alpha_depth", type=int )

        defaults['path'] = os.environ['DAE_NAME']
        defaults['nodes']="3153:12230"
        parser.add_argument(     "--path",    help="Path of geometry file  %(default)s",type=str)
        parser.add_argument("-g","--nodes",   help="DAENode.getall node(s) specifier %(default)s often 3153:12230 for some PMTs 5000:5100 ",type=str)

        defaults['size']="1440,852"
        defaults['frame'] = "1,1"
        parser.add_argument(     "--size",    help="Pixel size  %(default)s  small size 640,480", type=str)
        parser.add_argument(     "--frame",   help="Viewport framing  %(default)s",type=str)

        defaults['rgba'] = ".7,.7,.7,.5"
        defaults['vscale'] = 1.
        parser.add_argument(     "--rgba",     help="RGBA color of geometry, the alpha has a dramatic effect  %(default)s",type=str)
        parser.add_argument(     "--vscale",   help="Vertex scale, changing coordinate values stored in VBO. %(default)s", type=float)

        parser.set_defaults(**defaults)
        return parser, defaults

    def _make_live_parser(self, **kwa):
        with_defaults = kwa.pop('with_defaults',True)
        parser = ThrowingArgumentParser(**kwa)
        defaults = OrderedDict()

        # target based positioning mode switched on by presence of target 

        defaults['scaled_mode'] = False
        defaults['target'] = ".."
        defaults['jump'] = None
        defaults['ajump'] = None
        defaults['period'] = 1000
        defaults['eye'] = "-2,-2,0"
        defaults['look'] = "0,0,0"
        defaults['up'] = "0,0,1"
        defaults['norm'] = "0,0,0"
        defaults['fullscreen'] = False
        defaults['markers'] = False

        parser.add_argument( "--scaled-mode", action="store_true", help="In scaled mode the actual VBO vertex coordinates are scaled into -1:1, ie shrink world into unit cube. **FOR DEBUG ONLY** " )
        parser.add_argument("-t","--target",  help="[I] Node specification of solid on which to focus or empty string for all",type=str)
        parser.add_argument("-j","--jump",    help="[I] Animated transition to another node.")  
        parser.add_argument(     "--ajump",   help="[I] Append jump specs provided onto any existing ones.")  
        parser.add_argument(     "--period",   help="Animation interpolation frames to go from 0. to 1., %(default)s", type=float)  
        parser.add_argument("-e","--eye",     help="[I] Eye position ",type=str)
        parser.add_argument("-a","--look",    help="[I] Lookat position ",type=str)
        parser.add_argument("-u","--up",      help="[I] Up direction ",type=str)
        parser.add_argument( "--norm",    help="Dummy argument, used for informational output.",type=str)
        parser.add_argument( "--fullscreen", action="store_true", help="Start in fullscreen mode." )
        parser.add_argument( "--markers",   action="store_true", help="[I] Frustum and light markers." )
 
        # kernel switches
        defaults['cuda'] = False
        defaults['raycast'] = False
        parser.add_argument( "--cuda",      action="store_true", help="[I] Start in cuda mode." )
        parser.add_argument( "-r","--raycast",   action="store_true", help="[I] Raycast" )

        # kernel code
        defaults['kernel'] = "render_pbo"
        defaults['flags'] = "16,0"
        defaults['metric'] = "time"
        defaults['showmetric'] = False
        parser.add_argument( "--kernel", help="", type=str )
        parser.add_argument( "--flags", help="[I] g_flags constant provided to kernel, used for thread time presentation eg try 20,0  ", type=str  )
        parser.add_argument( "--metric", help="One of time/node/intersect/tri or default None", type=str  )
        parser.add_argument( "--showmetric", action="store_true", help="Switch on display of the metric and flags configured.")

        # kernel launch config, transitioning from 1D to 2D
        defaults['threads_per_block'] = 64  # 1D
        defaults['max_blocks'] = 1024       # 1D
        defaults['block'] = "16,16,1"       # 2D
        defaults['launch'] = "3,2,1"        # 2D
        parser.add_argument( "--threads-per-block", help="", type=int )
        parser.add_argument( "--max-blocks", help="", type=int )
        parser.add_argument( "--block", help="[I] String 3-tuple dimensions of the block of CUDA threads, eg \"32,32,1\" \"16,16,1\" \"8,8,1\" ", type=str  )
        parser.add_argument( "--launch", help="[I] String 3-tuple dimensions of the sequence of CUDA kernel launches, eg \"1,1,1\",  \"2,2,1\", \"2,3,1\" ", type=str  )

        # kernel params and how launched
        defaults['max_time'] = 3.0  ; MAX_TIME_WARN = "(greater than 4 seconds leads to GPU PANIC, GUI FREEZE AND SYSTEM CRASH) "
        defaults['allsync'] = True
        defaults['alpha_depth'] = 10
        parser.add_argument( "--allsync",   help="[I] always CUDA sync after each launch", action="store_true" )
        parser.add_argument( "--alpha-depth", help="[I] Chroma Raycaster alpha_depth", type=int )
        parser.add_argument( "--max-time", help="[I] Maximum time in seconds for kernel launch, if exceeded subsequent launches are ABORTed " + MAX_TIME_WARN , type=float )

        # camera
        defaults['near'] = 30.     
        defaults['far'] = 10000.  
        defaults['yfov'] = 50.
        defaults['nearclip'] = "0.0001,1000."
        defaults['farclip'] = "1,100000."
        defaults['yfovclip'] = "1.,179."
        parser.add_argument("-n","--near",      help="[I] Initial near in mm. %(default)s", type=float)
        parser.add_argument("--far",       help="[I] Initial far in mm. %(default)s", type=float)
        parser.add_argument("--yfov",      help="[I] Initial vertical field of view in degrees. %(default)s", type=float)
        parser.add_argument("--nearclip",  help="[I] Allowed range for near. %(default)s", type=str )
        parser.add_argument("--farclip",   help="[I] Allowed range for far. %(default)s", type=str )
        parser.add_argument("--yfovclip",  help="[I] Allowed range for yfov. %(default)s", type=str )

        # scene
        defaults['kscale'] = 100.
        defaults['parallel'] = False
        defaults['line'] = False
        defaults['fill'] = True
        defaults['transparent'] = True
        parser.add_argument("--kscale",    help="[I] Kludge scaling applied to MVP matrix. %(default)s", type=float)
        parser.add_argument("--parallel",                         action="store_true", help="Parallel projection, aka orthographic." )
        parser.add_argument("--line",         dest="line",        action="store_true",  help="Switch on line mode polygons  %(default)s" )
        parser.add_argument("--nofill",       dest="fill",        action="store_false", help="Inhibit fill mode polygons  %(default)s" )
        parser.add_argument("--notransparent",dest="transparent", action="store_false", help="Inhibit transparent fill  %(default)s" )

        # trackball  
        defaults['thetaphi'] = "0,0."
        defaults['xyz'] = "0,0,0"
        defaults['dragfactor'] = 1.
        defaults['trackballradius'] = 0.8
        defaults['translatefactor'] = 4000.
        parser.add_argument( "--thetaphi", help="Initial theta,phi. %(default)s", type=str)
        parser.add_argument( "--xyz", help="Initial viewpoint in canonical -1:1 cube coordinates %(default)s", type=str)
        parser.add_argument( "--dragfactor", help="Mouse/trackpad drag speed", type=float  )
        parser.add_argument( "--trackballradius", help="Trackball radius", type=float  )
        parser.add_argument( "--translatefactor", help="Scaling applied to trackball offset translations to conjure a trackball.xyz offset in camera frame.", type=float  )

        # lights
        defaults['light'] = True
        defaults['rlight'] = "-1,1,1"
        defaults['glight'] = "1,1,1"
        defaults['blight'] = "0,-1,1"
        defaults['flight'] = 1.
        defaults['wlight'] = 1.
        defaults['lights'] = "rgb"
        parser.add_argument("--nolight",      dest="light",       action="store_false", help="Inhibit light setup  %(default)s" )
        parser.add_argument("--rlight",  help="Red light position",type=str)
        parser.add_argument("--glight",  help="Green light position",type=str)
        parser.add_argument("--blight",  help="Blue light position",type=str)
        parser.add_argument("--flight",  help="Light position scale factor",type=float)
        parser.add_argument("--wlight",  help="Homogeonous 4th coordinate, 0 for infinity",type=float)
        parser.add_argument("--lights",  help="Enable rgb lights",type=str)
       
        if with_defaults:
            parser.set_defaults(**defaults)

        return parser, defaults



def check_live_parse( cfg ):
    cmdline = " ".join(sys.argv[1:])
    live_args = cfg(cmdline)
    print live_args



if __name__ == '__main__':
    cfg = DAEConfig(__doc__)
    cfg.init_parse()

    print "changed settings\n", cfg.changed_settings()
    print "all settings\n",cfg.all_settings()


 
