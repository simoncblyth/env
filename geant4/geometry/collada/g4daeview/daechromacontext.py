#!/usr/bin/env python
"""
DAEChromaContext
==================

To keep this usable from different environments, keep top level 
imports to a minimum. Especially ones that require contexts to be active.

For example DAERaycaster pulls in PixelBuffer which requires 
an active OpenGL context so defer the import until needed.

"""
import os, time, logging, traceback, json
import numpy as np

log = logging.getLogger(__name__)
json_ = lambda path:json.load(file(os.path.expandvars(path)))


try:
    import psutil 
except ImportError:
    psutil = None 


def min_max_step(wl):
    mn = wl.min()
    mx = wl.max()
    st = np.unique(np.diff(wl)).item()
    return mn,mx,st

def pycuda_init(gl=False):
    """
    Based on pycuda.gl.autoinit  pycuda.autoinit

    See :doc:`/env/pycuda/pycuda_memory`

    TODO: record htod geometry copy time
          and see if the below MAP_HOST flag makes a difference

    #import pycuda.gl.autoinit  # after this can use pycuda.gl.BufferObject(unsigned int)

    """
    log.debug("pycuda_init gl %s " % gl )
    import pycuda.driver as cuda

    if gl:
        import pycuda.gl as cudagl
    else:
        cudagl = None
    pass

    cuda.init()
    count = cuda.Device.count()
    assert count >= 1

    if gl:
        def _ctx_maker(dev):
            flags = cuda.ctx_flags.MAP_HOST
            log.debug("pycuda_init cudagl.make_context with flags %s " % flags )
            return cudagl.make_context(dev, flags)
    else:  
        def _ctx_maker(dev):
            flags = cuda.ctx_flags.MAP_HOST
            log.debug("pycuda_init non-gl make_context with flags %s " % flags )
            return dev.make_context(flags)


    from pycuda.tools import make_default_context
    global context
    context = make_default_context(_ctx_maker)
    device = context.get_device()

    def _finish_up():
        """
        Hmm this only gets done for clean exits 
        https://docs.python.org/2/library/atexit.html
        """
        print "_finish_up : cuda cleanup " 
        global context
        context.pop()
        context = None
        from pycuda.tools import clear_context_caches
        clear_context_caches()
    pass
    import atexit
    atexit.register(_finish_up)





def pick_seed():
    """Returns a seed for a random number generator selected using
    a mixture of the current time and the current process ID."""
    return int(time.time()) ^ (os.getpid() << 16)



class DAEMemoryMon(dict):
    def __init__(self):
        dict.__init__(self) 
        self._process = None

    def _get_process(self):
        if not psutil is None:
            if self._process is None:
                self._process = psutil.Process()
            pass
        pass
        return self._process
    process = property(_get_process)

    def __call__(self, tag):
        ps = self.process
        if ps is None:
            log.warn("no process memory monitoring as no psutil")
            return
        mem = ps.get_memory_info()
        MB = 1024*1024
        self["%s_vms" % tag ] = float(mem.vms)/MB
        self["%s_rss" % tag ] = float(mem.rss)/MB
        
    def metadata(self):
        memmon = self.copy()
        memmon['COLUMNS'] = ",".join(map(lambda k:"%s:f" % k, self))
        return memmon 



class Geant4MaterialMap(dict):
    def __init__(self, path="$G4DAECHROMA_CACHE_DIR/g4materials.json"):
        jsd = json_(path)
        keys = jsd.keys() 
        key = keys[0]
        dict.__init__(self, jsd[key])


class ChromaMaterialMap(dict):
    def __init__(self, chroma_geometry):
        self.material_count = len(chroma_geometry.unique_materials)
        for chindex, m in enumerate(chroma_geometry.unique_materials):
            g4name = m.name[:-9].replace("__","/")
            self[g4name] = chindex

    def write(self, path):
        log.info("writing to %s " % path )
        with open(path,"w") as fp:
            json.dump(self, fp) 

    def __str__(self):
        return "\n".join(["%s : %s " % (k, v ) for k, v in self.items()])



class DAEChromaContext(object):
    """
    DCC is intended as a rack on which to hang objects, 
    avoid "doing" anything substantial here 
    (eg do stepping in the propagator not here)
    """
    dummy = False
    def __init__(self, config, chroma_geometry, gl=0):
        log.debug("DAEChromaContext init, CUDA_PROFILE %s " % os.environ.get('CUDA_PROFILE',"not-defined") )
        config.args.gl = gl   # placeholder parameter

        self.config = config

        pycuda_init(gl=gl)
        self.chroma_geometry = chroma_geometry
        self.chroma_material_map = ChromaMaterialMap(chroma_geometry)   
        log.info("chroma_material_map : %s " % str(self.chroma_material_map))

        cmmpath = os.path.join(config.geocachefold, "ChromaMaterialMap.json")
        self.chroma_material_map.write(cmmpath)

        self.geant4_material_map = Geant4MaterialMap()   
        pass

        self.COLUMNS = 'hit:i,deviceid:i,gl:i,steps_per_call:i,threads_per_block:i,max_blocks:i,max_steps:i,seed:i,reset_rng_states:i,max_time:f'

        pass
        #self.setup_random_seed()
        pass
        self._gpu_seed = None
        self._gpu_geometry = None
        self._gpu_detector = None
        self._rng_states = None
        self._raycaster = None
        self._propagator = None
        self._parameters = None
        self._process = None


        self.mem = DAEMemoryMon()
        self.mem("init")

        log.debug("*** first GPU hit : creating gpu_detector  ")
        gpu_detector = self.gpu_detector
        self.metadata = gpu_detector.metadata 
        log.info("*** first GPU hit : done ")

    # first getters will invoke config_parameters resulting in configured values, 
    # for propagation level override call chroma.config_parameters(args, ctrl) to change  
    deviceid = property(lambda self:self.parameters['deviceid'])
    seed = property(lambda self:self.parameters['seed'])
    max_blocks = property(lambda self:self.parameters['max_blocks'])
    max_steps = property(lambda self:self.parameters['max_steps'])
    threads_per_block = property(lambda self:self.parameters['threads_per_block'])
    reset_rng_states = property(lambda self:self.parameters['reset_rng_states'])
    steps_per_call = property(lambda self:self.parameters['steps_per_call'])


    def incoming(self, request):
        self.mem("in")
        if hasattr(request, 'meta') and len(request.meta)>0:
            ctrl = request.meta[0].get('ctrl',{})
            args = request.meta[0].get('args',{})
            for meta in request.meta:
                log.info("incoming request with metadata keys : %s " % str(meta.keys()))
            pass
        else:
            log.warn("incoming request with no metadata, parameter defaults will be used") 
            ctrl = {}
            args = {}
        pass
        self.ctrl = ctrl
        self.args = args

        parameters = self.configure_parameters(ctrl, args, dump=True)
        pass
        if parameters['reset_rng_states']:
            log.warn("reset_rng_states")
            self.gpu_seed = parameters['seed']
        pass
        self.parameters = parameters


    def outgoing(self, response, results, extra=False):
        """
        :param response: NPL propagated photons
        :param results: dict of results from the propagation, eg times 
        """
        self.mem("out")
        metadata = {}
        metadata['parameters'] = self.parameters
        metadata['results'] = results
        metadata['ctrl'] = self.ctrl
        metadata['args'] = self.args
        if extra:
            metadata['geometry'] = self.gpu_detector.metadata
            metadata['cpumem'] = self.mem.metadata()
            metadata['chroma_material_map'] = self.chroma_material_map
            metadata['geant4_material_map'] = self.geant4_material_map
        pass
        response.meta = [metadata]
        return response


    def defaults(self):
        pairs = self.COLUMNS.split(",")
        atts = map(lambda pair:pair.split(':')[0], pairs)
        typs = map(lambda pair:pair.split(':')[1], pairs)
        vals = map(lambda att:getattr(self.config.args,att), atts)
        d = dict(zip(atts,vals))
        t = dict(zip(atts,typs))
        return d, t

    def _get_parameters(self):
        if self._parameters is None:
            log.warn("setting up default parameters : use configure_parameters to control ")
            self._parameters = self.configure_parameters(None, None, dump=True)
        return self._parameters 
    def _set_parameters(self, d):
        self._parameters = d
    parameters = property(_get_parameters, _set_parameters)


    def configure_parameters(self, ctrl, args, dump=True):
        """
        #. start with defaults from config/commandline
        #. apply overrides from ctrl and args
        """
        d, t = self.defaults()
        p = d.copy()

        def override(name, kv):
            if kv is None:return
            for k,v in kv.items(): 
                if k in p and v != p[k]:
                    log.warn("%s override  %s : %s -> %s " % (name, k, p[k], v))
                    p[k] = v 
                pass   
            pass

        def fixtype():
            for k in filter(lambda k:t[k] == 'i',p):
                try:
                    p[k] = int(p[k])
                except TypeError:
                    log.warn("type error for k %s d[k] %s " % (k,p[k])) 
                pass

        def pdump():
            log.info("default and ctrl override parameters")
            for k in p:
                print "[%s] %-30s : %10s : %10s " % (t[k], k, d[k], p[k])


        override('ctrl', ctrl)
        override('args', args)
        fixtype()

        if dump:
            pdump()

        p['COLUMNS'] = self.COLUMNS

        self._parameters = p
        return self._parameters 



    def setup_raycaster(self):
        from daeraycaster import DAERaycaster
        return DAERaycaster( self )

    def setup_propagator(self):
        from env.chroma.chroma_propagator.propagator import Propagator
        return Propagator( self )

    def setup_gpu_geometry(self):
        assert 0, "use setup_gpu_detector"
        from chroma.gpu.geometry import GPUGeometry
        assert self.chroma_geometry.__class__.__name__ == 'Detector', self.chroma_geometry.__class__.__name__
        return GPUGeometry( self.chroma_geometry)

    def setup_gpu_detector(self):
        """
        For add_pmt rather than add_solid which have a channel_id
        to copy onto the GPU 

        Use either gpu_geometry OR gpu_detector, NOT BOTH
        """
        from chroma.gpu.detector import GPUDetector
        assert self.chroma_geometry.__class__.__name__ == 'Detector', self.chroma_geometry.__class__.__name__

        standard_wavelengths = self.config.wavelengths
        assert len(standard_wavelengths) > 20, standard_wavelengths
        mn, mx, st = min_max_step(standard_wavelengths)
        log.debug("creating GPUGeometry using standard_wavelengths %s ==> %s:%s:%s " % (self.config.args.wavelengths, mn,mx,st)) 
        return GPUDetector( self.chroma_geometry, standard_wavelengths )

    def make_cuda_buffer_object(self, buffer_id ):
        import pycuda.gl as cuda_gl
        return cuda_gl.BufferObject(long(buffer_id))  

    def setup_rng_states(self):
        """
        Note that threads_per_block and max_block are properties 
        that can be changed via config_parameters
        """
        from chroma.gpu.tools import get_rng_states
        seed = self.gpu_seed 
        log.info("setup_rng_states using seed %s "  % seed )
        rng_states = get_rng_states(self.threads_per_block*self.max_blocks, seed=seed)
        return rng_states

    def setup_gpu_seed(self, seed):
        if seed is None:
            seed = pick_seed() 
            log.warn("RANDOMLY SETTING SEED TO %s " % seed )
            assert 0
        else:
            log.info("using seed %s " % seed )
        pass 
        np.random.seed(seed)
        return seed

    def _get_gpu_seed(self):
        """
        """
        if self._gpu_seed is None:
            assert 0, "use setter first"
            #self._gpu_seed = self.setup_gpu_seed(None)  
        return self._gpu_seed
    def _set_gpu_seed(self, seed):
        """
        This setter invalidates the RNG states, forcing recreation at next access, 
        invoke the setter with::

            chroma.gpu_seed = the-seed-integer

        """
        self._gpu_seed = self.setup_gpu_seed(seed)
        self._rng_states = None    
        #traceback.print_stack()
        pass
    gpu_seed = property(_get_gpu_seed, _set_gpu_seed)  

    def _get_rng_states(self):
        log.info("_get_rng_states")
        if self._rng_states is None:
            self._rng_states = self.setup_rng_states()
        return self._rng_states
    def _set_rng_states(self, rs):
        log.info("_set_rng_states")
        assert rs is None, "only allowed to set to None"
        self._rng_states = None
    rng_states = property(_get_rng_states, _set_rng_states, doc="setter accepts only None, to force recreation")
   




    def _get_gpu_geometry(self):
        if self._gpu_geometry is None:
            self._gpu_geometry = self.setup_gpu_geometry()
        return self._gpu_geometry
    gpu_geometry = property(_get_gpu_geometry)

    def _get_gpu_detector(self):
        if self._gpu_detector is None:
            self._gpu_detector = self.setup_gpu_detector()
        return self._gpu_detector
    gpu_detector = property(_get_gpu_detector)

    def _get_raycaster(self):
        if self._raycaster is None:
            self._raycaster = self.setup_raycaster()
        return self._raycaster
    raycaster = property(_get_raycaster)

    def _get_propagator(self):
        if self._propagator is None:
           self._propagator = self.setup_propagator()
        return self._propagator
    propagator = property(_get_propagator)  




if __name__ == '__main__':
    pass

