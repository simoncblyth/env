#!/usr/bin/env python

import logging
log = logging.getLogger(__name__)

from chroma.loader import load_geometry_from_string
from env.pycuda.pycuda_pyopengl_interop.pixel_buffer import PixelBuffer
from env.chroma.chroma_camera.pbo_renderer import PBORenderer

from render_pbo_config import Config


class Scene(object):
    """coordinator"""
    def __init__(self, config, trackball ):
        print config
        self.geometry = load_geometry_from_string(config.args.geometry)
        log.info("completed loading geometry from %s " % config.args.geometry)
        self.pixels = PixelBuffer(config.size, texture=True)
        log.info("created PixelBuffer %s  " % repr(config.size) )
        self.renderer = PBORenderer(self.pixels, self.geometry, config )
        log.info("created PBORenderer " )
        self.trackball = trackball

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


