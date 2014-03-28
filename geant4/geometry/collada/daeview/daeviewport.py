#!/usr/bin/env python

import logging
#import OpenGL.GL as gl
log = logging.getLogger(__name__)

class DAEViewport(object):
    """
    The values from the horses mouth depend on OpenGL
    state so what you get depends on precisely when you ask, they 
    seem wrong outdated initially and then correct later.

    But the values passed along with on_resize seem correct however, 
    presumably the dispatcher asked from the right juncture in OpenGL state.
    """
    def __init__(self, size ):
        self.width, self.height = size
        pass

    aspect = property(lambda self:float(self.width)/float(self.height))

    def resize(self, size ):
        self.width, self.height = size 
        log.info("resize %s " % self  )
  
    #def _get_height(self):
    #    viewport = gl.glGetIntegerv(gl.GL_VIEWPORT)
    #    return float(viewport[3])
    #_height = property(_get_height)
    #def _get_width(self):
    #    viewport = gl.glGetIntegerv(gl.GL_VIEWPORT)
    #    return float(viewport[2])
    #_width = property(_get_width)
    #def _get_aspect(self):
    #    viewport = gl.glGetIntegerv(gl.GL_VIEWPORT)
    #    return float(viewport[2])/float(viewport[3])
    #_aspect = property(_get_aspect)

    def __repr__(self):
        return "%s %s %s %s " % (self.__class__.__name__, self.width, self.height, self.aspect )



