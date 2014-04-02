#!/usr/bin/env python
"""

#. chroma needs separate geometry to the VBO used by daeviewgl.py 

Suspect will run out if space for full geometry when
have vertices duplicated for OpenGL and CUDA/Chroma::

    INFO:chroma:device usage:
    ----------
    nodes             2.8M  44.7M
    total                   44.7M
    ----------
    device total             2.1G
    device used              1.6G
    device free            516.1M


#. can Chroma be made to use the OpenGL vertices/faces VBO ?

Raycaster kernel spec
----------------------

#. chroma geometry intersection from chroma camera `render.cu`:
#. do not pass arrays of positions and rays to the kernel
#. pixel data lives on PBO
#. inputs to raycaster: 

   * eye, look, up
   * yfov, nx, ny, near  
   * or maybe the MVP matrix ?


#. need to fork chroma as easiest to modify render.cu in place


"""
import numpy as np
from pycuda import gpuarray as ga

from chroma import gpu
from chroma.loader import load_geometry_from_string

from chroma.log import logger, logging
logger.setLevel(logging.INFO)
log = logger

class DAERaycaster(object):
    def __init__(self, config ):
        geometry = load_geometry_from_string(config.args.path)
        self.geometry = geometry  
        self.size = config.size
        self.device_id = config.args.device_id
        self.max_alpha_depth = config.args.alpha_max
        self.alpha_depth = self.max_alpha_depth
        self.step = 1000
 
    npixels = property(lambda self:self.size[0]*self.size[1])

    def init_gpu(self):
        self.context = gpu.create_cuda_context(self.device_id)
        self.gpu_geometry = gpu.GPUGeometry(self.geometry)
        self.gpu_funcs = gpu.GPUFuncs(gpu.get_cu_module('mesh.h'))

        #
        #pos, dir = from_film(self.point, axis1=self.axis1, axis2=self.axis2,
        #                     size=self.size, width=self.film_width, focal_length=self.focal_length)
        #
        #self.rays = gpu.GPURays(pos, dir, max_alpha_depth=self.max_alpha_depth)
        #
        #self.pixels_gpu = ga.empty(self.npixels, dtype=np.uint32)
        #
 

    def run(self):
        #self.rays.render(self.gpu_geometry, self.pixels_gpu, self.alpha_depth, keep_last_render=False)
        self.context.pop()



if __name__ == '__main__':
    pass

    from daeconfig import DAEConfig
    config = DAEConfig(__doc__)
    config.init_parse()
    print config

    raycaster = DAERaycaster(config)
    raycaster.init_gpu()
    raycaster.run()


