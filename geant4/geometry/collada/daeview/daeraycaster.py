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

"""

import logging
log = logging.getLogger(__name__)

import numpy as np

import OpenGL.GL as gl
import OpenGL.GLUT as glut

from pycuda import gpuarray as ga

from chroma.loader import load_geometry_from_string
from chroma.gpu.geometry import GPUGeometry
from env.pycuda.pycuda_pyopengl_interop.pixel_buffer import PixelBuffer
from env.chroma.chroma_camera.pbo_renderer import PBORenderer


class DAERaycaster(object):
    def __init__(self, config, geometry ):
        self.config = config
        self.geometry = geometry 
        self.init_chroma(config)

    def init_chroma(self, config):
        if not config.args.with_chroma:
            return

        log.info("init_chroma")

        self.max_alpha_depth = config.args.max_alpha_depth
        self.alpha_depth = self.max_alpha_depth

        #self.chroma_geometry = self.geometry.make_chroma_geometry()   # this misses the bvh
        self.chroma_geometry = load_geometry_from_string(config.args.path)
        log.info("completed loading geometry from %s " % config.args.path)

        self.pixels = PixelBuffer(config.size, texture=True)
        log.info("created PixelBuffer %s  " % repr(config.size) )

        self.renderer = PBORenderer(self.pixels, self.chroma_geometry, config )
        log.info("created PBORenderer " )


    def render(self, pixel2world, eye):
        """
        """
        print "DAERaycaster"
        print "eye\n", eye
        print "pixel2world\n", pixel2world

        if not self.config.args.with_chroma:
            log.warn("chroma raycast rendering requires launch option: --with-chroma ")
            return 

        self.renderer.set_constants( self.pixels.size, eye, pixel2world )  # hmm changing size will cause problems
        self.renderer.render()
        self.pixels.draw()



if __name__ == '__main__':
    pass

    from daeconfig import DAEConfig
    config = DAEConfig(__doc__)
    config.init_parse()
    print config

    raycaster = DAERaycaster(config)
    raycaster.init_gpu()


