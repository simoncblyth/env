#!/usr/bin/env python

import logging
log = logging.getLogger(__name__)

from chroma.loader import load_geometry_from_string
from env.pycuda.pycuda_pyopengl_interop.pixel_buffer import PixelBuffer
from env.chroma.chroma_camera.pbo_renderer import PBORenderer
from env.geant4.geometry.collada.daeview.daegeometry import DAEGeometry

from render_pbo_config import Config


class Scene(object):
    """coordinator"""
    def __init__(self, config, trackball ):
        self.chroma_geometry = self.load_chroma_geometry( config )
        log.info("completed loading geometry from %s " % config.args.geometry)
        self.pixels = PixelBuffer(config.size, texture=True)
        log.info("created PixelBuffer %s  " % repr(config.size) )
        self.renderer = PBORenderer(self.pixels, self.chroma_geometry, config )
        self.trackball = trackball

    def load_chroma_geometry(self, config):
        #chroma_geometry = load_geometry_from_string(config.args.geometry)
        geometry = DAEGeometry( config.args.nodes )
        chroma_geometry = geometry.make_chroma_geometry()
        return chroma_geometry  

    def init(self):
        log.info("scene init")

    def draw(self):
        log.info("scene draw")
        self.trackball.push()

        self.renderer.render()
        self.pixels.draw()

        self.trackball.pop()


if __name__ == '__main__':
    pass


