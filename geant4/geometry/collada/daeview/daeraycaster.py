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
#from env.geant4.geometry.collada.collada_to_chroma import daeload


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

        self.size = config.size 
        self.chroma_geometry = self.geometry.make_chroma_geometry()  

        self.pixels = PixelBuffer( self.size, texture=True)
        log.info("created PixelBuffer %s  " % repr(config.size) )

        self.renderer = PBORenderer( self.pixels, self.chroma_geometry, config )
        log.info("created PBORenderer " )

    def resize(self, size ):
        log.info("DAERaycaster resize %s " % repr(size))
        if self.size == size:return
        self.pixels.resize(size)           
        self.renderer.resize(size)

    def render(self, view, camera, flags):
        if not self.config.args.with_chroma:
            log.warn("chroma raycast rendering requires launch option: --with-chroma ")
            return 

        log.info("DAERaycaster view %s camera %s flags %s " % (view, camera, flags))

        renderer = self.renderer

        renderer.origin = view.eye
        renderer.pixel2world = view.pixel2world_matrix( camera )
        renderer.flags = flags

        renderer.render(alpha_depth=self.config.args.alpha_depth, max_time=self.config.args.max_time)

        self.pixels.draw()


    def render_old(self, pixel2world, eye, flags):
        """
        """
        print "DAERaycaster"
        print "eye\n", eye
        print "pixel2world\n", pixel2world

        if not self.config.args.with_chroma:
            log.warn("chroma raycast rendering requires launch option: --with-chroma ")
            return 

        renderer = self.renderer

        renderer.origin = eye
        renderer.pixel2world = pixel2world 
        renderer.flags = flags

        renderer.render(alpha_depth=self.config.args.alpha_depth, max_time=self.config.args.max_time)


        self.pixels.draw()



if __name__ == '__main__':
    pass

    from daeconfig import DAEConfig
    config = DAEConfig(__doc__)
    config.init_parse()
    print config

    raycaster = DAERaycaster(config)
    raycaster.init_gpu()


