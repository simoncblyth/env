#!/usr/bin/env python
"""
DAEDirectConfig 
=================

"""
import os, logging, argparse, socket, md5, datetime, stat, shutil
import numpy as np
log = logging.getLogger(__name__)

try: 
    from collections import OrderedDict
except ImportError:
    OrderedDict = dict


digest_ = lambda _:md5.md5(_).hexdigest()
ivec_ = lambda _:map(int,_.split(","))
fvec_ = lambda _:map(float,_.split(","))


class DAEDirectConfig(object):
    """
    Non-GUI config, NB these parameters and defaults are also used
    for the GUI DAEConfig
    """
    block=property(lambda self:ivec_(self.args.block))
    launch=property(lambda self:ivec_(self.args.launch))
    flags=property(lambda self:ivec_(self.args.flags))

    def __init__(self, doc=""):
        """
        # these direct_ attributes allows DAEConfig to reuse and then 
        # expand on these parameter/default definitions
        """
        parser, defaults = self._make_direct_parser(doc)
        self.direct_parser = parser
        self.direct_defaults = defaults
        self._path = None
      
    def parse(self):
        self.direct_parser.set_defaults(**self.direct_defaults)
        args = self.direct_parser.parse_args()
        logging.basicConfig(level=getattr(logging, args.loglevel.upper()), format=args.logformat )
        self.args = args
 
    def _make_direct_parser(self, doc):
        parser = argparse.ArgumentParser(doc, add_help=False)
        defaults = OrderedDict()

        defaults['clargs'] = []
        defaults['loglevel'] = "INFO"
        defaults['ipython'] = False
        defaults['logformat'] = "%(asctime)-15s %(levelname)-7s %(name)-20s:%(lineno)-3d %(message)s"

        parser.add_argument( "clargs",nargs="*", help="Optional commandline args   %(default)s")  
        parser.add_argument( "--ipython",action="store_true", help="Drop into embedded ipython, where IPython.embed() is placed in the code. %(default)s")  
        parser.add_argument( "--loglevel",help="INFO/DEBUG/WARN/..   %(default)s")  
        parser.add_argument( "--logformat", help="%(default)s")  

        defaults['path'] = "dyb"
        #defaults['geometry']="3153:"
        #defaults['geometry']="2+,3153:"
        #defaults['geometry']="2+,3153:12221"  # skip the radslabs
        #defaults['geometry']="3153:12221"      # skip RPC and radslabs
        defaults['geometry'] = "DAE_GEOMETRY_%(path)s" 
        defaults['regexp'] = None
        defaults['geocache'] = True
        defaults['geocacheupdate'] = False
        defaults['geocachepath'] = None
        defaults['bound'] = True

        parser.add_argument("-p","--path",    help="Shortname indicating envvar DAE_NAME_SHORTNAME (or None indicating  DAE_NAME) that provides path to the G4DAE geometry file  %(default)s",type=str)
        parser.add_argument("-g","--geometry",   help="DAENode.getall node(s) specifier %(default)s often 3153:12230 for some PMTs 5000:5100 ",type=str)
        parser.add_argument(      "--regexp",   help="regexp search pattern eg PmtHemiCathode applied to node id that further restricts --geometry nodes",type=str)
        parser.add_argument(      "--nogeocache", dest="geocache", action="store_false", help="Save/load flattened geometry to/from binary npz cache. Default %(default)s." )
        parser.add_argument(      "--geocacheupdate", help="Remove geometry cache, to force rebuild. Default %(default)s.", action="store_true" )
        parser.add_argument(      "--geocachepath", help="Path to geometry cache. Default %(default)s." )
        parser.add_argument(     "--nobound",  dest="bound", action="store_false", help="Load geometry in pycollada unbound (local) coordinates, **FOR DEBUG ONLY** ")


        defaults['with_chroma'] = True 
        defaults['seed'] = 0

        parser.add_argument( "-C","--nochroma", dest="with_chroma", help="Indicate if Chroma is available.", action="store_false" )
        parser.add_argument( "--seed", help="Random Number seed, used for np.random.seed and curand setup", type=int  )


        parser.add_argument( "--zmqendpoint", help="Endpoint to for ZMQ ChromaPhotonList objects ", type=str  )
        parser.add_argument( "--zmqtunnelnode", help="Option interpreted at bash invokation level (not python) to specify remote SSH node to which a tunnel will be opened, strictly requires form `--zmqtunnelnode=N`  where N is an SSH config \"alias\".", type=str  )
        parser.add_argument( "--cuda-gdb", help="Option interpreted at bash invokation level (not python)", action="store_true")
        defaults['zmqendpoint'] = os.environ.get("ZMQ_BROKER_URL_BACKEND","tcp://localhost:5002")
        defaults['zmqtunnelnode'] = None
        defaults['cuda_gdb'] = False

        defaults['confdir'] = "~/.g4daeview/%(path)s"
        defaults['chroma_material_map'] = "chroma_material_map.json"
        defaults['chroma_surface_map'] = "chroma_surface_map.json"
        defaults['chroma_process_map'] = "chroma_process_map.json"

        parser.add_argument( "--confdir", help="Path to directory for config files such as bookmarks.  %(default)s", type=str  )
        parser.add_argument( "--chroma-material-map", help="Name of chroma material map file.  %(default)s", type=str  )
        parser.add_argument( "--chroma-surface-map", help="Name of chroma surface map file.  %(default)s", type=str  )
        parser.add_argument( "--chroma-process-map", help="Name of chroma process map file.  %(default)s", type=str  )

        defaults['wipegeometry'] = False
        parser.add_argument( "--wipegeometry", action="store_true", help="Wipe preexisting geometry maps before writing new ones, use this when changing geometry. Default %(default)s." )

        defaults['path_template'] = os.environ.get('DAE_PATH_TEMPLATE',None)
        defaults['key'] = 'CPL'

        parser.add_argument( "--key",   help="[I] ROOT Object Key to use with load/save. Default %(default)s.",type=str)
        parser.add_argument( "--path-template", help="Path template that load/save arguments fill in. Default %(default)s.",type=str)


        defaults['deviceid'] = None
        defaults['cuda_profile'] = False

        parser.add_argument(      "--device-id", help="CUDA device id.", type=str )
        parser.add_argument(      "--cuda-profile", help="Sets CUDA_PROFILE envvar.", action="store_true" )

        ## hmm these came from the live parser, moving them here means can no longer interactively (over udp) modify 
        # kernel launch config, transitioning from 1D to 2D
        defaults['deviceid'] = -1  # 1D
        defaults['threads_per_block'] = 64  # 1D
        defaults['max_blocks'] = 1024       # 1D
        defaults['max_steps'] = 30        
        defaults['block'] = "16,16,1"       # 2D
        defaults['launch'] = "3,2,1"        # 2D
        defaults['hit'] = True        
        defaults['reset_rng_states'] = True        
        defaults['gl'] = False       

        parser.add_argument( "--deviceid", help="For multiple GPU device selection when non negative", type=int )
        parser.add_argument( "--threads-per-block", help="", type=int )
        parser.add_argument( "--max-blocks", help="", type=int )
        parser.add_argument( "--max-steps", help="Maximum photon propagation steps. Default %(default)s", type=int )
        parser.add_argument( "--block", help="[I] String 3-tuple dimensions of the block of CUDA threads, eg \"32,32,1\" \"16,16,1\" \"8,8,1\" ", type=str  )
        parser.add_argument( "--launch", help="[I] String 3-tuple dimensions of the sequence of CUDA kernel launches, eg \"1,1,1\",  \"2,2,1\", \"2,3,1\" ", type=str  )

        parser.add_argument( "--nohit", dest="hit", action="store_false", help="Return only photons that hit sensitive detectors. ")
        parser.add_argument( "--noreset-rng-states", dest="reset_rng_states", action="store_false", help="Reset rng states for each propagation. ")
        parser.add_argument( "--nogl", dest="gl", action="store_false", help="Placeholder, set by argument to DAEChromaContext ")
        return parser, defaults 

    chroma_material_map = property(lambda self:self.resolve_confpath(self.args.chroma_material_map))
    chroma_surface_map = property(lambda self:self.resolve_confpath(self.args.chroma_surface_map))
    chroma_process_map = property(lambda self:self.resolve_confpath(self.args.chroma_process_map))

    def resolve_event_path(self, path_, subname=None):
        """ 
        Resolves paths to event files

        Using a path_template allows referencing paths in a
        very brief manner, ie with::
 
            export DAE_PATH_TEMPLATE="/usr/local/env/tmp/%s.root"

        Can use args `--load 1` 

        """
        if path_[0] == '/':return path_
        path_template = self.args.path_template
        if path_template is None:
            log.warn("path_template missing ")
            return path_
        log.debug("resolve_event_path path_template %s path_ %s " % (path_template, path_ ))  
        path = path_template % path_ 

        if subname is None:
            return path 
        else:
            base, ext = os.path.splitext(path) 
            return os.path.join( base, subname) 


    def resolve_path(self, path_):
        """
        Resolves paths to geometry files
        """
        pvar = "_".join(filter(None,["DAE_NAME",path_,]))
        pvar = pvar.upper()
        path = os.environ.get(pvar,None)
        log.debug("Using pvar %s to resolve path %s " % (pvar,path))
        assert not path is None, "Need to define envvar %s pointing to geometry file" % pvar
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


    def load_npl(self, name, key=None ):
        """
        """ 
        path = self.resolve_event_path(name)
        npl = np.load(path)
        return npl

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
    path = property(_get_path, doc="Resolves alias path arguments `-p dyb` via envvar `DAE_NAME_<DYB>`"  )

    def _get_confdir(self):
        path_ = self.args.path
        if path_ is None:
            path_ = ""
        return os.path.expanduser(os.path.expandvars( self.args.confdir % dict(path=path_) ))
    confdir = property(_get_confdir, doc="absolute path to confdir " )

    def resolve_confpath(self, name, timestamp=False):
        """
        Resolves path to config files, and creates directory if not existing

        For timestamp true the last change timestamp of a preexisting file 
        is incorporated into the name.  This allows prior bookmarks to be retained
        in timestamped files, to allow reverting to a prior set.

        :param name:
        :param timestamp:
        """
        path = os.path.join( self.confdir, name) 
        dir_ = os.path.dirname(path)
        if not os.path.exists(dir_):
            log.info("creating directory %s " % dir_ )
            os.makedirs(dir_) 
        pass
        if timestamp:
            if os.path.exists(path):
                stamp = datetime.datetime.fromtimestamp(os.stat(path)[stat.ST_CTIME]).strftime("%Y%m%d-%H%M")
                base, ext = os.path.splitext(name)        
                tname = "%s%s%s" % ( base, stamp, ext )        
                tpath = os.path.join(self, self.confdir, tname ) 
                return tpath
            else:
                log.warn("no preexisting path %s no need for timestamping  " % path ) 
        return path

    def _get_geocachepath(self):
        """
        Hmm would be better for the digest to be based on the list of 
        solids that the arguments (including regexp and geomerty) 
        leads to rather than the arguments 
        """
        gcp = self.args.geocachepath 
        if gcp is None:
            dig = digest_("%s%s" % (self.args.geometry, self.args.regexp))
            gcp = "%s.%s" % ( self.path, dig )
        return gcp
    geocachepath = property(_get_geocachepath) 

    def _get_chromacachepath(self):
        return os.path.join( self.geocachepath, "chroma_geometry")
    chromacachepath = property(_get_chromacachepath) 

    def wipe_geocache(self):
        cachedir = self.geocachepath
        if not os.path.exists(cachedir):
            return
        assert os.path.isdir(cachedir) 
        assert len(cachedir.split('/')) > 2, "cachedir sanitu check fail %s " % cachedir
        log.warn("removeing cachedir %s " % cachedir )
        shutil.rmtree(cachedir) 
 



if __name__ == '__main__':
    ddc = DAEDirectConfig(__doc__)
    ddc.parse()
    print ddc.args 


 
