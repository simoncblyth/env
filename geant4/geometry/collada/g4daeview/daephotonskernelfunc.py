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
        self.dphotons = dphotons
        self.ctx = ctx
        self.compile_kernel((("debug",debug),("max_slots",dphotons.data.max_slots),("numquad",dphotons.data.numquad)) )

    nphotons = property(lambda self:self.dphotons.data.nphotons)
    def compile_kernel(self, template_fill ):
        """
        #. compile kernel and extract __constant__ symbol addresses
        """
        log.info("compile_kernel %s %s " % (self.kernel_name, repr(template_fill)))
        module = get_cu_module(self.kernel_name, options=cuda_options, template_fill=template_fill )
        kernel = module.get_function( self.kernel_func )
        kernel.prepare(self.kernel_args)

        self.g_mask = module.get_global("g_mask")[0]  
        self._mask = None
        self.kernel = kernel

    def initialize_constants(self):
        self.mask = [-1,-1,-1,-1]

    def update_constants(self):
        self.mask = self.dphotons.param.kernel_mask

    def _get_mask(self):
        return self._mask 
    def _set_mask(self, mask):
        if mask == self._mask:return
        self._mask = mask
        log.info("_set_mask : memcpy_htod %s " % repr(mask))
        cuda_driver.memcpy_htod(self.g_mask, ga.vec.make_int4(*mask))
    mask = property(_get_mask, _set_mask, doc="setter copies to device __constant__ memory, getter returns cached value") 



if __name__ == '__main__':
    pass

