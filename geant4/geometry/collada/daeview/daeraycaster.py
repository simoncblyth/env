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
import numpy as np

import OpenGL.GL as gl
import OpenGL.GLUT as glut


from pycuda import gpuarray as ga

from chroma import gpu
from chroma.loader import load_geometry_from_string

#from chroma.log import logger, logging
#logger.setLevel(logging.INFO)
#log = logger

import logging
log = logging.getLogger(__name__)


class DAERaycaster(object):
    def __init__(self, config, camera, view, trackball ):
        self.config = config
        self.camera = camera
        self.view = view
        self.trackball = trackball

        self.init_chroma(config)

    def init_chroma(self, config):
        if not config.args.with_chroma:return
        pass
        self.chroma_geometry = load_geometry_from_string(config.args.path)
        self.device_id = config.args.device_id
        self.max_alpha_depth = config.args.alpha_max
        self.alpha_depth = self.max_alpha_depth
        self.step = 1000

    def init_gpu(self):
        self.context = gpu.create_cuda_context(self.device_id)
        self.gpu_geometry = gpu.GPUGeometry(self.chroma_geometry)
        self.gpu_funcs = gpu.GPUFuncs(gpu.get_cu_module('mesh.h'))

        #
        #pos, dir = from_film(self.point, axis1=self.axis1, axis2=self.axis2,
        #                     size=self.size, width=self.film_width, focal_length=self.focal_length)
        #
        #self.rays = gpu.GPURays(pos, dir, max_alpha_depth=self.max_alpha_depth)
        #
 
    def run(self):
        #self.rays.render(self.gpu_geometry, self.pixels_gpu, self.alpha_depth, keep_last_render=False)
        self.context.pop()

 
    def draw(self, kscale):
        camera = self.camera
        view = self.view

        scale = np.identity(4)   # it will be getting scaled down so have to scale it up, annoyingly 
        scale[0,0] = kscale
        scale[1,1] = kscale
        scale[2,2] = kscale

        pixel2camera_scaled = np.dot( scale, camera.pixel2camera ) # order matters, have to scale pixel2camera, not after camera2world
        camera2world = view.camera2world.matrix
        pixel2world = np.dot( camera2world, pixel2camera_scaled )   

        pixel_corners = camera.pixel_corners
        corners = np.array(pixel_corners.values())
        wcorners  = np.dot( corners, pixel2world.T )
        wcorners2 = np.dot( pixel2world, corners.T ).T   
        assert np.allclose( wcorners, wcorners2 )

        #indices = np.arange(0,camera.npixel,1024*16)            # cut down numbers of rays for visibility :w
        indices = np.random.randint(0, camera.npixel,1000)
        pixels = np.array(map(camera.pixel_xyzw, indices ))     # find a numpy way to map, if want to deal with all pixels
        wpoints = np.dot( pixels, pixel2world.T )

        eye = view.eye

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
    raycaster.run()


