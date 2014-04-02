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
   udp.py 


"""
import os, sys, logging, math
import argparse
from collections import OrderedDict
import numpy as np

log = logging.getLogger(__name__)

class ArgumentParserError(Exception): pass
class ThrowingArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ArgumentParserError(message)


class DAEConfig(object):
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
        
        logging.basicConfig(level=getattr(logging, args.loglevel), format="%(asctime)-15s %(message)s")
        np.set_printoptions(precision=4, suppress=True)
        self.args = args

    def live_parse(self, cmdline, post_process=False):
        live_args = None           
        try:
            live_args = self.live_parser.parse_args(cmdline.split(" "))
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
        fmt = " %-15s : %20s : %20s "
        return "\n".join([ fmt % (k,str(v)[:wid],str(getattr(args,k))[:wid]) for k,v in filter(filter_,defaults.items()) ])

    def base_settings(self, all_=False):
        return self._settings( self.args, self.base_defaults, all_ )

    def live_settings(self, all_=False):
        return self._settings( self.args, self.live_defaults, all_ )

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
        defaults['host'] = os.environ.get("DAEVIEW_UDP_HOST","127.0.0.1")
        defaults['port'] = os.environ.get("DAEVIEW_UDP_PORT", "15006")
        defaults['havecuda'] = True
        defaults['processor'] = "Invert"

        parser.add_argument("-l","--loglevel",help="INFO/DEBUG/WARN/..   %(default)s")  
        parser.add_argument( "-C","--nohavecuda", dest="havecuda", help="Inhibit use of cuda ", action="store_true"  )
        parser.add_argument(     "--processor", help="Name of the cuda processor to use.", type=str )

        parser.add_argument(   "--host", help="Hostname to bind to for UDP messages ", type=str  )
        parser.add_argument(   "--port", help="Port to bind to for UDP messages ", type=str  )


        defaults['path'] = os.environ['DAE_NAME']
        #defaults['nodes']="5000:5100"   # some PMTs for quick testing
        defaults['nodes']="3153:12230"

        parser.add_argument(     "--path",    help="Path of geometry file  %(default)s",type=str)
        parser.add_argument("-n","--nodes",   help="DAENode.getall node(s) specifier %(default)s",type=str)

        defaults['size']="1440,852"
        #defaults['size']="640,480"
        defaults['frame'] = "1,1"

        parser.add_argument(     "--size",    help="Pixel size  %(default)s", type=str)
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

        defaults['target'] = None
        defaults['jump'] = None
        defaults['ajump'] = None
        defaults['speed'] = 1e-3
        defaults['eye'] = "-2,-2,0"
        defaults['look'] = "0,0,0"
        defaults['up'] = "0,0,1"
        defaults['fullscreen'] = False
        defaults['cuda'] = False

        parser.add_argument("-t","--target",  help="[I] Node specification of solid on which to focus or empty string for all",type=str)
        parser.add_argument("-j","--jump",    help="[I] Animated transition to another node.")  
        parser.add_argument(     "--ajump",   help="[I] Append jump specs provided onto any existing ones.")  
        parser.add_argument(     "--speed",   help="Animation interpolatiom speed, %(default)s", type=float)  

        parser.add_argument("-e","--eye",     help="[I] Eye position ",type=str)
        parser.add_argument("-a","--look",    help="[I] Lookat position ",type=str)
        parser.add_argument("-u","--up",      help="[I] Up direction ",type=str)

        parser.add_argument(     "--fullscreen", action="store_true", help="Start in fullscreen mode." )
        parser.add_argument(     "--cuda",      action="store_true", help="[I] Start in cuda mode." )


        defaults['kscale'] = 100.
        defaults['near'] =   30.     
        defaults['far'] = 10000.  
        defaults['yfov'] = 50.
        defaults['nearclip'] = "0.0001,1000."
        defaults['farclip'] = "1,100000."
        defaults['yfovclip'] = "1.,179."
        defaults['parallel'] = False
        defaults['line'] = False
        defaults['fill'] = True
        defaults['transparent'] = True

        parser.add_argument("--kscale",    help="[I] Kludge scaling applied to MVP matrix. %(default)s", type=float)
        parser.add_argument("--near",      help="[I] Initial near in mm. %(default)s", type=float)
        parser.add_argument("--far",       help="[I] Initial far in mm. %(default)s", type=float)
        parser.add_argument("--yfov",      help="[I] Initial vertical field of view in degrees. %(default)s", type=float)
        parser.add_argument("--nearclip",  help="[I] Allowed range for near. %(default)s", type=str )
        parser.add_argument("--farclip",   help="[I] Allowed range for far. %(default)s", type=str )
        parser.add_argument("--yfovclip",  help="[I] Allowed range for yfov. %(default)s", type=str )
        parser.add_argument("--parallel",                         action="store_true", help="Parallel projection, aka orthographic." )
        parser.add_argument("--line",         dest="line",        action="store_true",  help="Switch on line mode polygons  %(default)s" )
        parser.add_argument("--nofill",       dest="fill",        action="store_false", help="Inhibit fill mode polygons  %(default)s" )
        parser.add_argument("--notransparent",dest="transparent", action="store_false", help="Inhibit transparent fill  %(default)s" )

        
        defaults['thetaphi'] = "0,0."
        defaults['xyz'] = "0,0,0"

        parser.add_argument(     "--thetaphi", help="Initial theta,phi. %(default)s", type=str)
        parser.add_argument(     "--xyz", help="Initial viewpoint in canonical -1:1 cube coordinates %(default)s", type=str)


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
       
     
        defaults['dragfactor'] = 1.
        defaults['trackballradius'] = 0.8
        defaults['translatefactor'] = 1000.

        parser.add_argument(   "--dragfactor", help="Mouse/trackpad drag speed", type=float  )
        parser.add_argument(   "--trackballradius", help="Trackball radius", type=float  )
        parser.add_argument(   "--translatefactor", help="Scaling applied to trackball offset translations to conjure a trackball.xyz offset in camera frame.", type=float  )

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


 
