#!/usr/bin/env python
"""
"""

import logging
log = logging.getLogger(__name__)

from env.pycuda.pycuda_pyopengl_interop.pixel_buffer import PixelBuffer
from env.chroma.chroma_camera.pbo_renderer import PBORenderer
from env.cuda.cuda_state import DriverState


class DAERaycaster(object):
    def __init__(self, ctx ):
        self.config = ctx.config
        self.size = self.config.size 
        self.pixels = PixelBuffer( self.size, texture=True)
        self.renderer = PBORenderer( self.pixels, ctx.gpu_detector, ctx.config )  # avoid double htod by using sub-instance gpu_detector instead of gpu_geometry

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

    def exit(self):
        import pycuda.driver as drv
        ds = DriverState(drv)
        log.info("DAERaycaster.exit : cuDriverState\n %s" % repr(ds) )




if __name__ == '__main__':
    pass

