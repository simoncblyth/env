#!/usr/bin/env python
"""

"""

import os
import numpy as np

import pycuda.driver as cuda
from pycuda import gpuarray as ga

from chroma.log import logger, logging
logger.setLevel(logging.INFO)
log = logger

from chroma import gpu
from chroma.loader import load_geometry_from_string

def floatarray_( a ):
    cpu = np.array(a).astype(np.float32)
    gpu = np.float32(cpu) 
    return gpu

def float4_( a ):
    cpu = np.array(a).astype(np.float32)
    gpu = np.array((cpu[0], cpu[1], cpu[2], cpu[3]), dtype=ga.vec.float4)
    return gpu

def float3_( a ):
    cpu = np.array(a).astype(np.float32)
    gpu = np.array((cpu[0], cpu[1], cpu[2]), dtype=ga.vec.float3)
    return gpu

def int2_( a ):
    cpu = np.array(a).astype(np.int32)
    gpu = np.array((cpu[0], cpu[1]), dtype=ga.vec.int2)
    return gpu


class Renderer(object):
    def __init__(self):
        pass
        device_id = None
        context = gpu.create_cuda_context(device_id)
        self.context = context 
        #self.geometry = load_geometry_from_string(os.environ['DAE_NAME'])
        #self.gpu_geometry = gpu.GPUGeometry(self.geometry)

    def prepare(self, origin, size, pixel2world ):
        module = gpu.get_cu_module('render_pbo.cu')

        d_origin = module.get_global("g_origin")  
        d_size   = module.get_global("g_size")  
        d_pixel2world = module.get_global("g_pixel2world")  

        cuda.memcpy_htod(d_origin,       float4_(origin))
        cuda.memcpy_htod(d_size,         int2_(size))
        cuda.memcpy_htod(d_pixel2world,  floatarray_(pixel2world))

    def launch(self):
        pass

    def cleanup(self):
        self.context.pop()



if __name__ == '__main__':
    rdr = Renderer()

    origin = (0,0,0,1)
    size = (640,480)
    p2w = np.identity(4) 

    rdr.prepare(origin, size, p2w )
    rdr.cleanup()



