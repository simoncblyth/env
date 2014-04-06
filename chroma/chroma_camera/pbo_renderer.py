#!/usr/bin/env python
"""
"""
import logging
log = logging.getLogger(__name__)

import numpy as np

from env.cuda.cuda_launch import Launch

import pycuda.gl as cuda_gl
import pycuda.driver as cuda
import pycuda.gpuarray as ga
import pycuda.gl.autoinit      # excludes use of non-gl autoinit

from chroma.gpu.geometry import GPUGeometry
from chroma.gpu.tools import get_cu_module, cuda_options


class PBORenderer(object):
    def __init__(self, pixels, chroma_geometry, config ):
        pass
        self.config = config

        npixels = pixels.npixels
        size = pixels.size
        self.pixels = pixels
        self.npixels = npixels

        #chroma_geometry.flatten()
        self.gpu_geometry = GPUGeometry( chroma_geometry )

        self.launch = Launch(npixels, threads_per_block=config.args.threads_per_block, max_blocks=config.args.max_blocks )

        #self.dxlen = ga.zeros(npixels, dtype=np.uint32)
        #dx_size = config.args.max_alpha_depth*npixels
        #self.dx    = ga.empty(dx_size, dtype=np.float32)
        #self.color = ga.empty(dx_size, dtype=ga.vec.float4)

        self.compile_()

        if hasattr(config, 'eye'): 
            self.set_constants(size, config.eye, config.pixel2world)

    def compile_(self):
        """
        #. compile kernel and extract __constant__ symbol addresses
        """
        module = get_cu_module('render_pbo.cu', options=cuda_options)
        #self.arg_format = "iiPPPPP"
        self.arg_format = "iiPP"

        self.g_size   = module.get_global("g_size")[0]  
        self.g_origin = module.get_global("g_origin")[0]
        self.g_pixel2world = module.get_global("g_pixel2world")[0]  

        kernel = module.get_function(self.config.args.kernel)
        kernel.prepare(self.arg_format)
        self.kernel = kernel

    def set_constants(self, size, origin, pixel2world ):
        """ copy constant values to GPU """

        log.info("set_constants")
        log.info("size %s " % repr(size))
        log.info("origin %s " % repr(origin))
        log.info("pixel2world %s " % repr(pixel2world))

        cuda.memcpy_htod(self.g_size,         ga.vec.make_int2(*size))
        cuda.memcpy_htod(self.g_origin,       ga.vec.make_float4(*origin))
        cuda.memcpy_htod(self.g_pixel2world,  np.float32(pixel2world))

    def check_args(self, arg_format, *args ):
        from pycuda._pvt_struct import pack
        i = 0
        for fmt,arg in zip(arg_format,args):
            print "checking arg %s %s %s " % (i,fmt, arg) 
            arg_buf = pack(fmt,arg)
            i += 1

    def render(self, alpha_depth=3, check_args=False):
        """
        :param alpha_depth:

        * http://stackoverflow.com/questions/6954487/how-to-use-the-prepare-function-from-pycuda

        """
        assert alpha_depth <= self.config.args.max_alpha_depth
        #if not keep_last_render:
        #    self.dxlen.fill(0)   # this is calling a gpuarray.fill kernel 

        pbo_mapping = self.pixels.cuda_pbo.map()

        args = [ np.uint32(alpha_depth), 
                 pbo_mapping.device_ptr(),
                 self.gpu_geometry.gpudata ]
        
        #         self.dx.gpudata, 
        #         self.dxlen.gpudata, 
        #         self.color.gpudata ]

        if check_args:
            self.check_args( self.arg_format[1:], *args)  # skip offset arg

        block = self.launch.block
        calls = 0 
        print "pixels %s launch %s " % (repr(self.pixels.size), repr(self.launch))
        for offset, count, blocks_per_grid in self.launch.chunker:
            grid=(blocks_per_grid, 1)
            print "[%s] offset %s grid %s block %s " % (calls, offset, repr(grid),repr(block))
            self.kernel.prepared_call( grid, block, np.uint32(offset), *args )
            if self.config.args.allsync:
                try:
                    cuda.Context.synchronize()  
                except:
                    print "sync exception" 
            pass
            calls += 1
        pass
        cuda.Context.synchronize()  # OMITTING THIS SYNC CAUSES AN UNRECOVERABLE GUI FREEZE
        pbo_mapping.unmap()



