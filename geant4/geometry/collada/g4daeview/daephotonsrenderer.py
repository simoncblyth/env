#!/usr/bin/env python

import OpenGL.GL as gl
import OpenGL.GLUT as glut

from daephotonsshader import DAEPhotonsShader
from daevertexbuffer import DAEVertexBuffer


class DAEPhotonsRenderer(object):
    def __init__(self, dphotons):
        self.dphotons = dphotons
        self.shader = DAEPhotonsShader()

    lbuffer = property(lambda self:self.dphotons.lbuffer)
    pbuffer = property(lambda self:self.dphotons.pbuffer)

    def create_buffer(self, data, indices ):
        if data is None or indices is None:
            return None
        return DAEVertexBuffer( data, indices  )
        
    def draw(self):
        """
        qcut restricts elements drawn, the default of 1 corresponds to all

        Formerly used separate point and line VBOs with point drawn with::

             self.pvbo.draw(mode=gl.GL_POINTS, what='pc', count=qcount  , offset=0 )

        The below succeeds to draws points at start and end of the lines::

             self.lvbo.draw(mode=gl.GL_POINTS,  what='pc', count=2*qcount, offset=0, att=1 )  
      
        Attempts to use `offset=1` to draw the endpoint do not cause error but fail to 
        draw the end point, seeming just drawing the startpoint.  Is this a misunderstanding 
        or a bug ? 

        Its a misunderstanding the glDrawElements offset is offsetting applied to
        the entire indices array, ie it controls where to start getting indices from.
        For offsets within each element have to use VertexAttrib offsets.

        """ 
        qcount = self.dphotons.qcount

        gl.glPointSize(self.dphotons.param.fphopoint)  

        self.lbuffer.draw(mode=gl.GL_LINES,   what='pc', count=2*qcount, offset=0, att=1 )
        self.lbuffer.draw(mode=gl.GL_POINTS,  what='pc', count=qcount,   offset=0, att=2 )     # draw start point

        self.shader.bind()
        self.lbuffer.draw(mode=gl.GL_POINTS,  what='pc', count=qcount,   offset=0, att=3 )    # draw the endpoint 
        self.shader.unbind()

        gl.glPointSize(1)  

if __name__ == '__main__':
    pass


