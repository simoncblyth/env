#!/usr/bin/env python

import logging
log = logging.getLogger(__name__)

import OpenGL
OpenGL.FORWARD_COMPATIBLE_ONLY = True    

import OpenGL.GL as gl
import OpenGL.GLUT as glut

from daephotonsshader import DAEPhotonsShader
from daevertexbuffer import DAEVertexBuffer
from daephotonskernel import DAEPhotonsKernel


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
        """
        self.dphotons = dphotons
        self.chroma = chroma
        self.interop = not self.chroma.dummy
        self.kernel = DAEPhotonsKernel(dphotons) if self.interop else None 
        self.shader = DAEPhotonsShader(dphotons) 
        self.invalidate_buffers()
        self.count = 0 

    def invalidate_buffers(self):
        """
        Called when changes (eg loading a new event or changing a selection)
        invalidate the current buffers.  This will force buffer recreation on 
        next usage (eg when drawing).
        """
        self._pbuffer = None
        self._lbuffer = None

    def _get_pbuffer(self):
        if self._pbuffer is None:
           self._pbuffer = self.create_buffer(self.dphotons.data)  
        return self._pbuffer
    pbuffer = property(_get_pbuffer, doc="point buffer, without doubling : used with geometry shader to generate 2nd vertices and lines ")  

    def _get_lbuffer(self):
        if self._lbuffer is None:
           self._lbuffer = self.create_buffer(self.dphotons.data)  
        return self._lbuffer
    lbuffer = property(_get_lbuffer, doc="line buffer, with doubled vertices : not used when geometry shader available")  

    def create_buffer(self, data ):
        """
        #. buffer creation does not belong in DAEPhotonsData as OpenGL specific
        """
        self.count += 1
        log.warn("############ create_buffer [count %s]  ##################### %s " % (self.count, repr(data.data.dtype)) )
        vbo = DAEVertexBuffer( data.data, data.indices, force_attribute_zero=data.force_attribute_zero, shader=self.shader  )
        self.interop_gl_to_cuda(vbo)
        return vbo

    def interop_cuda_to_gl(self, buf ):
        """
        Ends CUDA responsibility, allows OpenGL access for drawing 
        """
        if not self.interop:return
        buf.cuda_buffer_object.unregister() 

    def interop_gl_to_cuda(self, buf ):
        """ 
        Registering the VBO with CUDA, by creation of cuda_gl BufferObject 
        """
        if not self.interop:return
        buf.cuda_buffer_object = self.chroma.make_cuda_buffer_object(buf.vertices_id)

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

    def interop_call_cuda_kernel(self, buf):
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

        #. modify Chroma propagation to use the VBO, not just simple DAEPhotonsKernel

        #. add mask/bits uniforms and flags attribute 

           * use these to move selection logic onto GPU, controlled by setting mask/bits uniform
           * apply selection either by geometry shader omission or color alpha control 

        #. try to generate something for the polarization too ?

           * GL_LINE_STRIP comes out of the geometry shader pipe, 
             stick polz direction compass on the end ?

        #. aiming towards once only buffer creation, ie only when a new ChromaPhotonList is loaded


        """
        qcount = self.dphotons.qcount
        gl.glPointSize(self.dphotons.param.fphopoint)  
       
        self.kernel.update_constants()   # can polling param changes be avoided ?

        # interop sharing of pbuffer 
        self.interop_call_cuda_kernel(self.pbuffer)
        self.interop_cuda_to_gl(self.pbuffer)
        self.pbuffer.draw(mode=gl.GL_POINTS,  what='', count=qcount,   offset=0, att=1 ) 
        self.interop_gl_to_cuda(self.pbuffer)

        gl.glPointSize(1)  



if __name__ == '__main__':
    pass


