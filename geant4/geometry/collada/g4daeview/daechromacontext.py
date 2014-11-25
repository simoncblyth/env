#!/usr/bin/env python
"""
DAEChromaContext
==================

To keep this usable from different environments, keep top level 
imports to a minimum. Especially ones that require contexts to be active.

For example DAERaycaster pulls in PixelBuffer which requires 
an active OpenGL context so defer the import until needed.

"""
import os, time, logging
log = logging.getLogger(__name__)

import numpy as np


def pycuda_init(gl=False):
    """
    Based on pycuda.gl.autoinit  pycuda.autoinit

    See :doc:`/env/pycuda/pycuda_memory`

    TODO: record htod geometry copy time
          and see if the below MAP_HOST flag makes a difference

    #import pycuda.gl.autoinit  # after this can use pycuda.gl.BufferObject(unsigned int)

    """
    log.info("pycuda_init gl %s " % gl )
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
            # ? how to set flags for gl ?
            return cudagl.make_context(dev)
    else:  
        def _ctx_maker(dev):
            flags = cuda.ctx_flags.MAP_HOST
            log.info("pycuda_init _ctx_maker with flags %s " % flags )
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
        pass

        self.COLUMNS = 'hit:i,deviceid:i,gl:i,threads_per_block:i,max_blocks:i,max_steps:i,seed:i,reset_rng_states:i'
        self.deviceid = config.args.deviceid 

        # temporarily pinned here 
        self.threads_per_block = config.args.threads_per_block
        self.max_blocks = config.args.max_blocks
        self.seed = config.args.seed

        #self.max_steps = config.args.max_steps
        #self.reset_rng_states = 1      # reset rng_states for every propagation, to repeat same random sequence
        #self.propagatorcode = propagatorcode

        pass
        #self.setup_random_seed()
        pass
        self._gpu_seed = None
        self._gpu_geometry = None
        self._gpu_detector = None
        self._rng_states = None
        self._raycaster = None
        self._propagator = None

        log.info("*** first GPU hit : creating gpu_detector  ")
        gpu_detector = self.gpu_detector
        self.metadata = gpu_detector.metadata  
        log.info("*** first GPU hit : done ")



    def defaults(self):
        pairs = self.COLUMNS.split(",")
        atts = map(lambda pair:pair.split(':')[0], pairs)
        typs = map(lambda pair:pair.split(':')[1], pairs)
        vals = map(lambda att:getattr(self.config.args,att), atts)
        d = dict(zip(atts,vals))
        t = dict(zip(atts,typs))
        return d, t

    def parameters(self, ctrl, args, dump=True):
        """
        #. start with defaults from config/commandline
        #. apply overrides from ctrl and args
        """
        d, t = self.defaults()

        def override(name, kv):
            if kv is None:return
            for k,v in kv.items(): 
                if k in d and v != d[k]:
                    log.warn("%s override  %s : %s -> %s " % (name, k,d[k], v))
                    d[k] = v 
                pass   
            pass

        override('ctrl', ctrl)
        override('args', args)

        for k in filter(lambda k:t[k] == 'i',d):
            try:
                d[k] = int(d[k])
            except TypeError:
                log.warn("type error for k %s d[k] %s " % (k,d[k])) 
            pass

        if dump:
            log.info("default and ctrl override parameters")
            for k in d:
                print "[%s] %-30s : %10s : %10s " % (t[k], k, d[k], p[k])

        d['COLUMNS'] = self.COLUMNS
        return d

    def setup_raycaster(self):
        from daeraycaster import DAERaycaster
        return DAERaycaster( self )

    def setup_propagator(self):
        from env.chroma.chroma_propagator.propagator import Propagator
        return Propagator( self )

    def setup_gpu_geometry(self):
        from chroma.gpu.geometry import GPUGeometry
        assert self.chroma_geometry.__class__.__name__ == 'Detector', self.chroma_geometry.__class__.__name__
        return GPUGeometry( self.chroma_geometry )

    def setup_gpu_detector(self):
        """
        For add_pmt rather than add_solid which have a channel_id
        to copy onto the GPU 

        Use either gpu_geometry OR gpu_detector, NOT BOTH
        """
        from chroma.gpu.detector import GPUDetector
        assert self.chroma_geometry.__class__.__name__ == 'Detector', self.chroma_geometry.__class__.__name__
        return GPUDetector( self.chroma_geometry )

    def make_cuda_buffer_object(self, buffer_id ):
        import pycuda.gl as cuda_gl
        return cuda_gl.BufferObject(long(buffer_id))  

    def setup_rng_states(self):
        """
        Hmm this placement prevents ctrl variation of nthreads_per_block 
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

