#!/usr/bin/env python
"""
"""
import logging
log = logging.getLogger(__name__)

import numpy as np
from env.cuda.cuda_launch import Launch2D

#import pycuda.gl as cuda_gl
import pycuda.driver as cuda
import pycuda.gpuarray as ga

# hmm isnt this done elsewhere in DAEChromaContext ? maybe needed for standalone test
import pycuda.gl.autoinit      # excludes use of non-gl autoinit

from chroma.gpu.tools import get_cu_module, cuda_options


class PBORenderer(object):
    def __init__(self, pixels, gpu_geometry, config ):
        """
        :param pixels:
        :param gpu_geometry: `GPUGeometry` instance
        :param config:
        """
        self.pixels = pixels
        self.gpu_geometry = gpu_geometry 
        self.config = config
        self.launch = Launch2D( work=pixels.size, launch=config.launch, block=config.block )
        self.cudacheck = getattr(config, 'cudacheck', None)
        self.times = []

        self._alpha_depth = None
        self._offset = None
        self._flags = None
        self._size = None
        self._origin = None
        self._pixel2world = None

        # non-GPU resident constants  
        self.max_time = config.args.max_time
        self.allsync = config.args.allsync
        self.showmetric = config.args.showmetric
        self._prior_flags = None

        template_uncomment = None if config.args.metric is None else (('metric',config.args.metric),) 
        self.compile_kernel(template_uncomment=template_uncomment)
        self.initialize_gpu_constants()
        self.config_showmetric()

    def __repr__(self):
        eye =  "%.2f,%.2f,%.2f" % tuple(self.origin[:3])
        return eye + " " + repr(self.launch) + " " + ",".join(map(lambda _:"%.2f" % _, self.times))  

    def resize(self, size):
        log.debug("PBORenderer resize %s " % repr(size))
        if self.size == size:return   # getter returns cached local value
        self.size = size              # setter copies to device
        self.launch.resize(size)

    def config_showmetric(self):
        """
        TODO: tidy up this mess of flags 

        Maybe eliminate some moving parts by regarding external message 
        re-config as changing the DAEConfig ?
        """
        if not self.showmetric:
            if not self.flags == (0,0):
                self._prior_flags = self.flags
            pass
            self.flags = 0,0
            log.debug("switch OFF showmetric, but retain any flag settings aleady configured" ) 
        else:
            if self.flags != (0,0):
                log.info("switch ON showmetric using already set self.flags %s " % repr(self.flags))
            elif self._prior_flags is None: 
                self.flags = self.config.flags 
                log.info("switch ON showmetric using self.config.flags %s " % repr(self.config.flags))
            else:
                self.flags = self._prior_flags 
                log.info("switch ON showmetric using self._prior_flags %s " % repr(self._prior_flags))
        pass

    def reconfig(self, **kwa):
        self.launch.reconfig(**kwa) 

        # mix of non-gpu and gpu residents 
        for qty in ("max_time","allsync","flags","alpha_depth","showmetric",):
            if qty in kwa:
                setattr(self,qty,kwa[qty])
        pass 
        self.config_showmetric()

    def initialize_gpu_constants(self):
        """
        # these setters copy to GPU device __constant__ memory
        """
        self.offset = (0,0)
        self.size = self.pixels.size
        self.flags = 0,0
        self.alpha_depth = self.config.args.alpha_depth

        if hasattr(self.config, 'eye'): 
            self.origin = self.config.eye
            self.pixel2world = self.config.pixel2world
        else:
            log.debug("initializing GPU constants origin/pixel2world to defaults, need to be set appropriately for geometry to see anything  ") 
            self.origin = (0,0,0,1)
            self.pixel2world = np.identity(4)

    def compile_kernel(self, template_uncomment = None):
        """
        #. compile kernel and extract __constant__ symbol addresses
        """
        module = get_cu_module('render_pbo.cu', options=cuda_options, template_uncomment=template_uncomment)

        self.g_alpha_depth  = module.get_global("g_alpha_depth")[0]  
        self.g_offset  = module.get_global("g_offset")[0]  
        self.g_flags   = module.get_global("g_flags")[0]  
        self.g_size   = module.get_global("g_size")[0]  
        self.g_origin = module.get_global("g_origin")[0]
        self.g_pixel2world = module.get_global("g_pixel2world")[0]  

        kernel = module.get_function(self.config.args.kernel)
        kernel.prepare("PP")

        self.kernel = kernel


    def _get_max_time(self):
        return self._max_time
    def _set_max_time(self, max_time):
        assert max_time <= 4.5
        self._max_time = max_time
    max_time = property(_get_max_time, _set_max_time)


    def _get_alpha_depth(self):
        return self._alpha_depth
    def _set_alpha_depth(self, alpha_depth):
        if alpha_depth == self._alpha_depth:return
        assert alpha_depth <= self.config.args.max_alpha_depth
        self._alpha_depth = alpha_depth
        cuda.memcpy_htod(self.g_alpha_depth, np.uint32(alpha_depth))
    alpha_depth = property(_get_alpha_depth, _set_alpha_depth) 

    def _get_offset(self):
        return self._offset 
    def _set_offset(self, offset):
        if offset == self._offset:return
        self._offset = offset
        cuda.memcpy_htod(self.g_offset,         ga.vec.make_int2(*offset))
    offset = property(_get_offset, _set_offset) 

    def _get_flags(self):
        return self._flags 
    def _set_flags(self, flags):
        if flags == self._flags:return
        self._flags = flags
        cuda.memcpy_htod(self.g_flags,         ga.vec.make_int2(*flags))
    flags = property(_get_flags, _set_flags) 

    def _get_size(self):
        return self._size 
    def _set_size(self, size):
        if size == self._size:return
        self._size = size
        cuda.memcpy_htod(self.g_size,         ga.vec.make_int2(*size))
    size = property(_get_size, _set_size) 
      
    def _get_origin(self):
        return self._origin
    def _set_origin(self, origin):
        #if origin == self._origin:return
        self._origin = origin
        cuda.memcpy_htod(self.g_origin,       ga.vec.make_float4(*origin))
    origin = property(_get_origin, _set_origin) 
 
    def _get_pixel2world(self):
        return self._pixel2world
    def _set_pixel2world(self, pixel2world):
        #if pixel2world == self._pixel2world:return
        self._pixel2world = pixel2world
        cuda.memcpy_htod(self.g_pixel2world,  np.float32(pixel2world))
    pixel2world = property(_get_pixel2world, _set_pixel2world) 



    def render(self):
        #log.info("render %s " % repr(self.launch))
        pbo_mapping = self.pixels.cuda_pbo.map()

        args = [ pbo_mapping.device_ptr(), self.gpu_geometry.gpudata ]

        times = []
        abort = False
        for launch, work, offset, grid, block in self.launch.iterator:
            self.offset = offset 
            if abort:
                t = -1
            else:
                get_time = self.kernel.prepared_timed_call( grid, block, *args )
                t = get_time()
                if t > self.max_time:
                    abort=True
                    log.warn("kernel launch time %s > max_time %s  , ABORTING RENDER " % (t, self.max_time) )
            times.append(t)
            if self.allsync:
                cuda.Context.synchronize()  
            pass
        pass

        cuda.Context.synchronize()  # OMITTING THIS SYNC CAN CAUSE AN IRRECOVERABLE GUI FREEZE
        pbo_mapping.unmap()

        times.append(sum(times))
        self.times = times

        if self.cudacheck is not None:
            self.cudacheck.parse_profile()
            self.cudacheck.compare_with_launch_times(times, self.launch)
        else:  
            log.info(repr(self))


if __name__ == '__main__':
    pass


