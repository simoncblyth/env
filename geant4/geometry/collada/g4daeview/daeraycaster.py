#!/usr/bin/env python
"""
"""

import logging
log = logging.getLogger(__name__)
from env.pycuda.pycuda_pyopengl_interop.pixel_buffer import PixelBuffer
from env.chroma.chroma_camera.pbo_renderer import PBORenderer

class DAERaycaster(object):
    def __init__(self, config, geometry ):
        self.config = config
        self.size = config.size 
        chroma_geometry = geometry.make_chroma_geometry()  
        self.pixels = PixelBuffer( self.size, texture=True)
        self.renderer = PBORenderer( self.pixels, chroma_geometry, config )

    def resize(self, size ):
        log.debug("DAERaycaster resize %s " % repr(size))
        if self.size == size:return
        self.pixels.resize(size)           
        self.renderer.resize(size)

    def reconfig(self, **kwa):
        log.debug("DAERaycaster reconfig %s " % repr(kwa))
        self.renderer.reconfig( **kwa )

    def render(self):
        self.renderer.origin = self.transform.eye
        self.renderer.pixel2world = self.transform.pixel2world
        self.renderer.render()
        self.pixels.draw()


if __name__ == '__main__':
    pass

