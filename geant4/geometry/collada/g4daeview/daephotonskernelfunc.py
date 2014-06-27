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
        
        template_fill = (("max_slots",dphotons.data.max_slots),
                         ("numquad",dphotons.data.numquad),
                         ("debugphoton",dphotons.config.args.debugphoton),
                         ) 
        template_uncomment = (("debug",debug),) 
        self.compile_kernel( template_fill, template_uncomment )

    nphotons = property(lambda self:self.dphotons.data.nphotons)
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

        # must not be defaults otherwise setter will not memcopy_htod
        self._mask = [-99,-99,-99,-99]  
        self._anim = [-99,-99,-99,-99]

        self.kernel = kernel

    def initialize_constants(self):
        assert 0

    def update_constants(self):
        kernel_mask = self.dphotons.param.kernel_mask
        #log.info("update_constants %s " % repr(kernel_mask))
        self.mask = kernel_mask
        #self.anim = self.dphotons.param.kernel_anim

    def _get_mask(self):
        return self._mask 
    def _set_mask(self, mask):
        if mask == self._mask:return
        self._mask = mask
        log.info("_set_mask : memcpy_htod %s " % repr(mask))
        cuda_driver.memcpy_htod(self.g_mask, ga.vec.make_int4(*mask))
    mask = property(_get_mask, _set_mask, doc="mask: setter copies to device __constant__ memory, getter returns cached value") 

    def _get_anim(self):
        return self._anim 
    def _set_anim(self, anim):
        #log.info("_set_anim %s " % anim )
        if anim == self._anim:
            #log.info("_set_anim %s no change %s " % (anim, self._anim) )
            return
        self._anim = anim
        #log.info("_set_anim : memcpy_htod %s " % repr(anim))
        cuda_driver.memcpy_htod(self.g_anim, ga.vec.make_float4(*anim))
    anim = property(_get_anim, _set_anim, doc="anim: setter copies to device __constant__ memory, getter returns cached value") 

    def _get_time(self):
        return self._anim[0]
    def _set_time(self, time):
        #log.info("_set_time %s " % time )
        anim = self._anim[:]
        anim[0] = time
        self.anim = anim
    time = property(_get_time, _set_time)

    def _get_cohort(self):
        return self._anim[1:4]
    def _set_cohort(self, cohort):
        cohort = map(float,cohort.split(",")) 
        assert len(cohort) == 3 
        log.info("_set_cohort %s " % repr(cohort))
        anim = self._anim[:]
        anim[1:4] = cohort
        self.anim = anim
    cohort = property(_get_cohort, _set_cohort)




if __name__ == '__main__':
    pass

