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
import os, logging, argparse, socket, md5
import numpy as np
from configbase import ConfigBase, ThrowingArgumentParser


digest_ = lambda _:md5.md5(_).hexdigest()


try: 
    from collections import OrderedDict
except ImportError:
    OrderedDict = dict

log = logging.getLogger(__name__)

def address():
    """
    Not a general solution, but working for me 

    http://stackoverflow.com/questions/166506/finding-local-ip-addresses-using-pythons-stdlib
    """
    return socket.gethostbyname(socket.gethostname())


ivec_ = lambda _:map(int,_.split(","))
fvec_ = lambda _:map(float,_.split(","))

class DAEConfig(ConfigBase):

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

    def __init__(self, doc=""):
        ConfigBase.__init__(self, doc) 
        self._path = None

    def resolve_event_path(self, path_ ):
        """ 
        Resolves paths to event files

        Using a path_template allows referencing paths in a
        very brief manner, ie with::
 
            export DAE_PATH_TEMPLATE="/usr/local/env/tmp/%(arg)s.root"

        Can use args `--load 1` 

        """
        if path_[0] == '/':return path_
        path_template = self.args.path_template
        if path_template is None:
            log.warn("path_template missing ")
            return path_
        log.debug("resolve_event_path path_template %s path_ %s " % (path_template, path_ ))  
        path = path_template % { 'arg':path_ }
        return path 

    def resolve_path(self, path_):
        """
        Resolves paths to geometry files
        """
        pvar = "_".join(filter(None,["DAE_NAME",path_,]))
        pvar = pvar.upper()
        path = os.environ.get(pvar,None)
        log.debug("Using pvar %s to resolve path %s " % (pvar,path))
        assert not path is None, "Need to define envvar pointing to geometry file"
        assert os.path.exists(path), path
        return path

    def load_cpl(self, name, key=None ):
        """
        Requires envvar from cpl-;cpl-export to find the ROOT library 
        """ 
        if key is None:
            key = self.args.key
        from env.chroma.ChromaPhotonList.cpl import load_cpl
        path = self.resolve_event_path(name)
        cpl = load_cpl(path, key )
        return cpl

    def save_cpl(self, cpl, name, key=None ):
        """
        Requires envvar from cpl-;cpl-export to find the ROOT library 
        """ 
        if key is None:
            key = self.args.key
        path = self.resolve_event_path(name)
        from env.chroma.ChromaPhotonList.cpl import save_cpl
        save_cpl( path, key, cpl )

    def _get_path(self):
        if self._path is None:
            self._path = self.resolve_path(self.args.path)
        return self._path
    path = property(_get_path)

    def _get_bookmarks(self):
        path_ = self.args.path
        if path_ is None:
            path_ = ""
        pass
        return self.args.bookmarks % dict(path=path_)
    bookmarks = property(_get_bookmarks, doc="bookmark file name incorporating the geometry file shortname")  

    def _get_geocachepath(self):
        gcp = self.args.geocachepath 
        if gcp is None:
            gcp = "%s.%s.npz" % ( self.path, digest_(self.args.geometry) )
        return gcp
    geocachepath = property(_get_geocachepath) 

    def _get_timerange(self):
        timerange = self.args.timerange
        return None if timerange is None else fvec_(timerange)
    def _set_timerange(self, timerange):
        self.args.timerange = timerange
    timerange = property(_get_timerange, _set_timerange) 


    def _make_base_parser(self, doc):
        """
        Base parser handles arguments/options that 
        must be set at initialisation, either because they 
        only make sense to be done there or due to 
        handling of live updates not being implemented.
        """
        parser = argparse.ArgumentParser(doc, add_help=False)

        defaults = OrderedDict()

        defaults['clargs'] = []
        defaults['loglevel'] = "INFO"
        defaults['logformat'] = "%(asctime)-15s %(name)-20s:%(lineno)-3d %(message)s"
        defaults['legacy'] = False


        defaults['debugkernel'] = False
        defaults['debugpropagate'] = True
        defaults['debugphoton'] = 0

        defaults['prescale'] = 1
        defaults['max_slots'] = 10
        defaults['host'] = os.environ.get("DAEVIEW_UDP_HOST","127.0.0.1")
        defaults['port'] = os.environ.get("DAEVIEW_UDP_PORT", "15006")
        defaults['address'] = address()
        defaults['seed'] = 0
        defaults['bookmarks'] = "bookmarks_%(path)s.cfg"
        defaults['zmqendpoint'] = os.environ.get("ZMQ_BROKER_URL_BACKEND","tcp://localhost:5002")
        defaults['zmqtunnelnode'] = None

        parser.add_argument( "clargs",nargs="*", help="Optional commandline args   %(default)s")  
        parser.add_argument( "--loglevel",help="INFO/DEBUG/WARN/..   %(default)s")  
        parser.add_argument( "--logformat", help="%(default)s")  
        parser.add_argument( "--legacy", dest="legacy", action="store_true", help="Sets `legacy=True`, with `color` and `position` rather than custom OpenGL attributes, default %(default)s." )
        parser.add_argument( "--debugshader", action="store_true", help="Use debug shader without geometry stage, default %(default)s." )
        parser.add_argument( "--debugkernel", action="store_true", help="Enables VBO_DEBUG in propagate_vbo.cu, default %(default)s." )
        parser.add_argument( "--debugpropagate", action="store_true", help="Readback propagated VBO into numpy array and persist to propagated.npz, also compares with any same seed prior files. Default %(default)s." )
        parser.add_argument( "--debugphoton", type=int, help="photon_id to debug in propagate_vbo.cu when --debugkernel is enabled, default %(default)s." )
        parser.add_argument( "--prescale", help="Scale down photon array sizes yieled by DAEPhotonsData by subsampling, default %(default)s.", type=int )
        parser.add_argument( "--max-slots", dest="max_slots", help="Blow up photon array and VBO sizes to hold multiple parts of the propagation, default %(default)s.", type=int )
        parser.add_argument( "--host", help="Hostname to bind to for UDP messages ", type=str  )
        parser.add_argument( "--port", help="Port to bind to for UDP messages ", type=str  )
        parser.add_argument( "--address", help="IP address %(default)s", type=str  )
        parser.add_argument( "--seed", help="Random Number seed, used for np.random.seed and curand setup", type=int  )
        parser.add_argument( "--bookmarks", help="Path to persisted bookmarks  %(default)s", type=str  )
        parser.add_argument( "--zmqendpoint", help="Endpoint to for ZMQ ChromaPhotonList objects ", type=str  )
        parser.add_argument( "--zmqtunnelnode", help="Option interpreted at bash invokation level (not python) to specify remote SSH node to which a tunnel will be opened, strictly requires form `--zmqtunnelnode=N`  where N is an SSH config \"alias\".", type=str  )

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

        defaults['path'] = "dyb"
        #defaults['geometry']="3153:"
        #defaults['geometry']="2+,3153:"
        #defaults['geometry']="2+,3153:12221"  # skip the radslabs
        defaults['geometry']="3153:12221"      # skip RPC and radslabs
        defaults['geocache'] = False
        defaults['geocachepath'] = None

        defaults['bound'] = True
        parser.add_argument("-p","--path",    help="Shortname indicating envvar DAE_NAME_SHORTNAME (or None indicating  DAE_NAME) that provides path to the G4DAE geometry file  %(default)s",type=str)
        parser.add_argument("-g","--geometry",   help="DAENode.getall node(s) specifier %(default)s often 3153:12230 for some PMTs 5000:5100 ",type=str)
        parser.add_argument(      "--geocache", help="Save/load flattened geometry to/from binary npz cache. Default %(default)s.", action="store_true" )
        parser.add_argument(      "--geocachepath", help="Path to geometry cache. Default %(default)s." )
        parser.add_argument(     "--nobound",  dest="bound", action="store_false", help="Load geometry in pycollada unbound (local) coordinates, **FOR DEBUG ONLY** ")

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
        defaults['object'] = None
        defaults['jump'] = None
        defaults['ajump'] = None
        defaults['period'] = 1000
        defaults['timeperiod'] = 300 
        defaults['timerange'] = None
        defaults['eye'] = "-1,-1,0"  # -2,-2,0 formerly 
        defaults['look'] = "0,0,0"
        defaults['up'] = "0,0,1"
        defaults['norm'] = "0,0,0"
        defaults['fullscreen'] = False
        defaults['markers'] = False

        parser.add_argument( "--scaled-mode", action="store_true", help="In scaled mode the actual VBO vertex coordinates are scaled into -1:1, ie shrink world into unit cube. **FOR DEBUG ONLY** " )
        parser.add_argument("-t","--target",  help="[I] Node specification of solid on which to focus or empty string for all",type=str)
        parser.add_argument(     "--object",  help="[I] Specification of event object on which to focus",type=str)
        parser.add_argument("-j","--jump",    help="[I] Animated transition to another node.")  
        parser.add_argument(     "--ajump",   help="[I] Append jump specs provided onto any existing ones.")  
        parser.add_argument(     "--period",   help="Animation interpolation frames to go from 0. to 1., %(default)s", type=float)  
        parser.add_argument(     "--timeperiod",   help="Time Animation interpolation frames to go from 0. to 1., %(default)s", type=float)  
        parser.add_argument(     "--timerange",   help="Comma delimited timerange in nanoseconds, eg 0,100 (0.2997 m/ns) %(default)s", type=str)  
        parser.add_argument("-e","--eye",     help="[I] Eye position ",type=str)
        parser.add_argument("-a","--look",    help="[I] Lookat position ",type=str)
        parser.add_argument("-u","--up",      help="[I] Up direction ",type=str)
        parser.add_argument( "--norm",    help="Dummy argument, used for informational output.",type=str)
        parser.add_argument( "--fullscreen", action="store_true", help="Start in fullscreen mode." )
        parser.add_argument( "--markers",   action="store_true", help="[I] Frustum and light markers." )
        parser.add_argument( "--mode", help="Photon style mode, default %(default)s.", type=int )


        # event
        defaults['style'] = "noodles"
        defaults['live'] = True
        defaults['load'] = None
        defaults['save'] = None
        defaults['saveall'] = False
        defaults['key'] = 'CPL'
        defaults['path_template'] = os.environ.get('DAE_PATH_TEMPLATE',None)
        #defaults['pholine']  = False
        #defaults['phopoint']  = True
        defaults['fpholine'] = 100.
        defaults['fphopoint'] = 2
        defaults['time'] = 0.
        defaults['cohort'] = "-1,-1,-1"
        defaults['qcut'] = 1.
        defaults['tcut'] = 0.
        defaults['mask'] = -1
        defaults['bits'] = -1
        defaults['pid'] = -1
        defaults['mode'] = -1
        defaults['reload']  = False

        parser.add_argument( "--style", help="Key controlling photon render eg confetti/spagetti/movie/.., identifying shaders (vertex/geometry/fragment) and rendering techniques to use, default %(default)s." )
        parser.add_argument( "--nolive",  dest="live", help="[I] Disable live updating via ZMQRoot messages. Default %(default)s.", action="store_false")
        parser.add_argument( "--live",    dest="live", help="[I] Enable live updating via ZMQRoot messages. Default %(default)s.", action="store_true")
        parser.add_argument( "--load",  help="[I] Path to .root file to read, eg containing ChromaPhotonList instances. Default %(default)s.",type=str)
        parser.add_argument( "--save",  help="[I] Path to .root file to write. Default %(default)s.",type=str)
        parser.add_argument( "--saveall",  help="[I] Save all CPL received. Default %(default)s.", action="store_true")
        parser.add_argument( "--key",   help="[I] ROOT Object Key to use with load/save. Default %(default)s.",type=str)
        parser.add_argument( "--path-template", help="Path template that load/save arguments fill in. Default %(default)s.",type=str)
        parser.add_argument( "--fpholine", help="In --pholine mode controls line length from position to position + momdirection*fpho. Default %(default)s.",type=float)
        parser.add_argument( "--fphopoint", help="Present photons as points of size fphopoint. Default %(default)s.",type=float)
        parser.add_argument( "--time", help="Time used for photon history animation. Default %(default)s.",type=float)
        parser.add_argument( "--cohort", help="Comma delimited cohort start/end/mode with times in ns Default %(default)s.",type=str)
        #parser.add_argument( "--pholine", help="Present photons as lines from position to position + momdirection*fpho. Default %(default)s.",action="store_true")
        #parser.add_argument( "--nopholine", dest="pholine", help="Switch off line representation, returning to point. %(default)s.",action="store_false")
        #parser.add_argument( "--phopoint", help="Present photons as points of size fphopoint. Default %(default)s.",action="store_true")
        parser.add_argument( "--qcut", help="Select photons to present based on quantity count, in range 0. to 1., where 1. means all. Default %(default)s.",type=float)
        parser.add_argument( "--tcut", help="Select photons to present based on their global time, in range 0. to 1., where 1. means all. Default %(default)s.",type=float)

        parser.add_argument( "--mask", help="Apply mask bitwise AND selection to status flags of Chroma stepped photons. Default %(default)s", type=str )  
        parser.add_argument( "--bits", help="Apply history bits equality selection to flags of Chroma stepped photons. Default %(default)s", type=str )  
        parser.add_argument( "--pid",  help="Photon ID Selection. Default %(default)s", type=str )  
        parser.add_argument( "--reload",  help="[I] Reload current loaded event, useful after stepping. Default %(default)s.", action="store_true")


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
    

    print cfg.path

 
