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
        
    def former_draw(self):
        """
        Using single doubled up lbuffer with VertexAttrib offset tricks 
        (via the att=1,2,3) to pull out lines and points from that  
        """
        qcount = self.dphotons.qcount
        gl.glPointSize(self.dphotons.param.fphopoint)  

        self.lbuffer.draw(mode=gl.GL_LINES,   what='pc', count=2*qcount, offset=0, att=1 )
        self.lbuffer.draw(mode=gl.GL_POINTS,  what='pc', count=qcount,   offset=0, att=2 )     # startpoint
        #self.lbuffer.draw(mode=gl.GL_POINTS,  what='pc', count=qcount,   offset=0, att=3 )    # endpoint 

        gl.glPointSize(1)  

    def draw(self):
        """
        Non-doubled pbuffer with geometry shader is used to generate the 2nd vertex 
        (based on momdir attribute) and line primitive on the device

        NEXT: 

        #. add line coloring by wavelength computed on device
        #. add mask/bits uniforms and flags attribute 

           * use these to move selection logic onto GPU, controlled by setting mask/bits uniform
           * apply selection either by geometry shader omission or color alpha control 

        #. try to generate something for the polarization too ?

           * GL_LINE_STRIP comes out of the geometry shader pipe, 
             stick polz direction compass on the end ?

        #. aiming towards once only buffer creation, ie only when a new ChromaPhotonList is loaded
        #. do the interop dance to get CUDA/Chroma to make propagation changes 
           inside the one-and-only OpenGL buffer

        """
        qcount = self.dphotons.qcount
        gl.glPointSize(self.dphotons.param.fphopoint)  

        self.pbuffer.draw(mode=gl.GL_POINTS,  what='pc', count=qcount,   offset=0, att=1 )    # points

        self.shader.bind()
        self.pbuffer.draw(mode=gl.GL_POINTS,  what='pc', count=qcount,   offset=0, att=1, program=self.shader.shader.program)    # lines via geometry shader
        self.shader.unbind()

        gl.glPointSize(1)  

    


if __name__ == '__main__':
    pass


