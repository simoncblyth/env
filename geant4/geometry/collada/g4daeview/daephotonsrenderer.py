#!/usr/bin/env python

import logging
log = logging.getLogger(__name__)

import OpenGL
OpenGL.FORWARD_COMPATIBLE_ONLY = True    

import OpenGL.GL as gl
import OpenGL.GLUT as glut

from daephotonskernel import DAEPhotonsKernel
from daephotonsshader import DAEPhotonsShader
from daevertexbuffer import DAEVertexBuffer


class DAEPhotonsRenderer(object):
    """
    Coordinates presentation of photons with sharing of OpenGL VBO between:

    #. OpenGL buffer drawing 
    #. OpenGL GLSL shading, including geometry shader to generate 
       lines primitives from points primitives together with a momentum direction
    #. PyCUDA modification of the VBO content

    """
    def __init__(self, dphotons, chroma ):
        """
        :param dphotons: DAEPhotons instance
        :param chroma: chroma context instance

        #. the debug shader skips geometry shader just drawing end-points 

        The source of the kernel and shaders depends on the data structure, so 
        use string interpolation to tie these together somewhat.
        """
        self.chroma = chroma
        self.dphotons = dphotons

        shader = DAEPhotonsShader(dphotons) 
        print shader

        kernel = DAEPhotonsKernel(dphotons) if self.interop else None
        print kernel

        self.shader = shader
        self.kernel = kernel

    interop = property(lambda self:not self.chroma.dummy)
    lbuffer = property(lambda self:self.dphotons.lbuffer)
    pbuffer = property(lambda self:self.dphotons.pbuffer)

    def create_buffer(self, data, indices, force_attribute_zero ):
        if data is None or indices is None:
            return None
        pass 
        log.debug("create_buffer ")
        #print data

        vbo = DAEVertexBuffer( data, indices, force_attribute_zero=force_attribute_zero  )
        self.interop_gl_to_cuda(vbo)
        return vbo

    def interop_cuda_to_gl(self, buf ):
        if self.interop:
            buf.cuda_buffer_object.unregister() 

    def interop_gl_to_cuda(self, buf ):
        if self.interop:
            buf.cuda_buffer_object = buf.make_cuda_buffer_object(self.chroma)

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

    def interop_call(self, buf):
        """
        NEXT:
 
        #. variable qcount ? avoid that  
        """ 
        if self.kernel is None:return
        import pycuda.driver as cuda_driver

        buf_mapping = buf.cuda_buffer_object.map()
        self.kernel( buf_mapping.device_ptr(), self.dphotons.qcount )   
        cuda_driver.Context.synchronize()
        buf_mapping.unmap()

    def draw(self):
        """
        Non-doubled pbuffer with geometry shader is used to generate the 2nd vertex 
        (based on momdir attribute) and line primitive on the device

        NEXT: 

        #. modify Chroma propagation to use the VBO, not just 
           simple DAEPhotonsKernel

        #. add line coloring by wavelength computed on device
        #. add mask/bits uniforms and flags attribute 

           * use these to move selection logic onto GPU, controlled by setting mask/bits uniform
           * apply selection either by geometry shader omission or color alpha control 

        #. try to generate something for the polarization too ?

           * GL_LINE_STRIP comes out of the geometry shader pipe, 
             stick polz direction compass on the end ?

        #. aiming towards once only buffer creation, ie only when a new ChromaPhotonList is loaded

        DONE:

        #. interop dance to get CUDA to make changes inside the OpenGL VBO 

        """
        qcount = self.dphotons.qcount
        gl.glPointSize(self.dphotons.param.fphopoint)  
        
        self.interop_call(self.pbuffer)
        self.interop_cuda_to_gl(self.pbuffer)


        self.pbuffer.draw(mode=gl.GL_POINTS,  what='', count=qcount,   offset=0, att=1, shader=self.shader )    # lines via geometry shader


        # hmm draw without shader, get rid of this
        self.pbuffer.draw(mode=gl.GL_POINTS,  what='pc', count=qcount,   offset=0, att=1 )    # points


        gl.glPointSize(1)  

        self.interop_gl_to_cuda(self.pbuffer)



if __name__ == '__main__':
    pass


