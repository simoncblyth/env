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


#. Ray directions for each pixel are trivial in eye space



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


    def illustrate(self, pixel2world, eye, camera ):
        """
        :param pixel2world: matrix represented by 4x4 numpy array 
        :param eye: world frame eye coordinates, typically from view.eye
        :param camera: DAECamera instance, used to get pixel counts, corner pixel coordinates
                       and to provide pixel coordinates for pixel indices 
        """
        corners = np.array(camera.pixel_corners.values())
        wcorners = np.dot( corners, pixel2world.T )           # world frame corners

        #wcorners2 = np.dot( pixel2world, corners.T ).T    pre/post matrix multiplication equivalent when transpose appropriately
        #assert np.allclose( wcorners, wcorners2 )

        indices = np.random.randint(0, camera.npixel,1000)    # random pixel indices 
        pixels = np.array(map(camera.pixel_xyzw, indices ))   # find a numpy way to map, if want to deal with all pixels
        wpoints = np.dot( pixels, pixel2world.T )             # world frame random pixels


        gl.glDisable( gl.GL_LIGHTING )
        gl.glDisable( gl.GL_DEPTH_TEST )
        gl.glColor3f( 0.,0.,1. ) 

        gl.glPointSize(10)
        gl.glBegin( gl.GL_POINTS )
        for wcorner in wcorners:
            gl.glVertex3f( *wcorner[:3] )
            pass 
        gl.glEnd()

        for wcorner in wcorners:
            gl.glBegin( gl.GL_LINES )
            gl.glVertex3f( *eye[:3] )
            gl.glVertex3f( *wcorner[:3] )
            gl.glEnd()
            pass 

        for wpoint in wpoints:
            gl.glBegin( gl.GL_LINES )
            gl.glVertex3f( *eye[:3] )
            gl.glVertex3f( *wpoint[:3] )
            gl.glEnd()
            pass 

        gl.glEnable( gl.GL_LIGHTING )
        gl.glEnable( gl.GL_DEPTH_TEST )



if __name__ == '__main__':
    pass

    from daeconfig import DAEConfig
    config = DAEConfig(__doc__)
    config.init_parse()
    print config

    raycaster = DAERaycaster(config)
    raycaster.init_gpu()


