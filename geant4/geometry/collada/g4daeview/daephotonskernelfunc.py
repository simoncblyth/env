#!/usr/bin/env python
"""
"""

import logging
log = logging.getLogger(__name__)

import numpy as np

import pycuda.driver as cuda_driver
import pycuda.gpuarray as ga

from chroma.gpu.tools import get_cu_module, cuda_options, chunk_iterator, to_float3

class DAEPhotonsKernelFunc(object):
    max_time = 4.
    def __init__(self, dphotons, ctx, debug=1):
        """
        :param dphotons: DAEPhotons instance
        :param ctx: `DAEChromaContext` instance, for GPU config and geometry
        """
        log.debug("%s debug %s " % (self.__class__.__name__, debug ))
        self.dphotons = dphotons
        self.ctx = ctx
        
        template_fill = (("max_slots",dphotons.param.max_slots),
                         ("numquad",dphotons.param.numquad),
                         ("debugphoton",dphotons.config.args.debugphoton),
                         ) 
        template_uncomment = (("debug",debug),) 
        self.compile_kernel( template_fill, template_uncomment )

    nphotons = property(lambda self:len(self.dphotons.photons))
    def compile_kernel(self, template_fill, template_uncomment=None):
        """
        #. compile kernel and extract __constant__ symbol addresses
        """

        log.debug("compile_kernel %s template_fill %s template_uncomment %s " % (self.kernel_name, repr(template_fill),repr(template_uncomment)))
        module = get_cu_module(self.kernel_name, options=cuda_options, template_fill=template_fill, template_uncomment=template_uncomment)
        kernel = module.get_function( self.kernel_func )
        kernel.prepare(self.kernel_args)

        self.g_mask = module.get_global("g_mask")[0]  
        self.g_anim = module.get_global("g_anim")[0]  
        self.g_mate = module.get_global("g_mate")[0]  
        self.g_mode = module.get_global("g_mode")[0]  
        self.g_surf = module.get_global("g_surf")[0]  

        dummy = [-99,-99,-99,-99]  # must not be defaults otherwise setter will not memcopy_htod
        self._mask = dummy
        self._anim = dummy
        self._mate = dummy
        self._mode = dummy
        self._surf = dummy

        self.kernel = kernel

    def initialize_constants(self):
        assert 0

    def update_constants(self):
        kernel_mask = self.dphotons.param.kernel_mask
        #log.info("update_constants %s " % repr(kernel_mask))
        self.mask = kernel_mask
        #self.anim = self.dphotons.param.kernel_anim


    def _get_surface(self):
        return self._surf 
    def _set_surface(self, surf):
        if surf == self._surf:return
        self._surf = surf
        log.debug("_set_surf : memcpy_htod %s " % repr(surf))
        cuda_driver.memcpy_htod(self.g_surf, ga.vec.make_int4(*surf))
    surface = property(_get_surface, _set_surface, doc="surf: setter copies to device __constant__ memory, getter returns cached value") 


    def _get_material(self):
        return self._mate 
    def _set_material(self, mate):
        if mate == self._mate:return
        self._mate = mate
        log.debug("_set_mate : memcpy_htod %s " % repr(mate))
        cuda_driver.memcpy_htod(self.g_mate, ga.vec.make_int4(*mate))
    material = property(_get_material, _set_material, doc="mate: setter copies to device __constant__ memory, getter returns cached value") 


    def _get_mode(self):
        return self._mode
    def _set_mode(self, mode):
        mode = map(int,mode.split(","))
        if mode == self._mode:return
        self._mode = mode
        log.debug("_set_mode : memcpy_htod %s " % repr(mode))
        cuda_driver.memcpy_htod(self.g_mode, ga.vec.make_int4(*mode))
    mode = property(_get_mode, _set_mode )

    def _get_mask(self):
        return self._mask 
    def _set_mask(self, mask):
        if mask == self._mask:return
        self._mask = mask
        log.debug("_set_mask : memcpy_htod %s " % repr(mask))
        cuda_driver.memcpy_htod(self.g_mask, ga.vec.make_int4(*mask))
    mask = property(_get_mask, _set_mask, doc="mask: setter copies to device __constant__ memory, getter returns cached value") 

    def _get_anim(self):
        return self._anim 
    def _set_anim(self, anim):
        if anim == self._anim:return
        self._anim = anim
        cuda_driver.memcpy_htod(self.g_anim, ga.vec.make_float4(*anim))
    anim = property(_get_anim, _set_anim, doc="anim: setter copies to device __constant__ memory, getter returns cached value") 

    def _get_time(self):
        return self._anim[0]
    def _set_time(self, time):
        anim = self._anim[:]
        anim[0] = time
        self.anim = anim
    time = property(_get_time, _set_time)

    def _get_cohort(self):
        return self._anim[1:4]
    def _set_cohort(self, cohort):
        cohort = map(float,cohort.split(",")) 
        assert len(cohort) == 3 
        log.debug("_set_cohort %s " % repr(cohort))
        anim = self._anim[:]
        anim[1:4] = cohort
        self.anim = anim
    cohort = property(_get_cohort, _set_cohort)




if __name__ == '__main__':
    pass

