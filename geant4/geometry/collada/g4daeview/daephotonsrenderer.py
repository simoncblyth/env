#!/usr/bin/env python

import logging
log = logging.getLogger(__name__)

import OpenGL.GL as gl

from daevertexbuffer import DAEVertexBuffer
from daephotonsshader import DAEPhotonsShader
from daephotonskernel import DAEPhotonsKernel
from daephotonspropagator import DAEPhotonsPropagator


class DAEPhotonsRenderer(object):
    """
    Coordinates presentation of photons with sharing of OpenGL VBO between:

    #. OpenGL buffer drawing 
    #. OpenGL GLSL shading
    #. PyCUDA modification of the VBO content

    Features
    ~~~~~~~~~ 

    #. selection of photons to render is controlled by mask/bits CUDA constants, 
       which are used in CUDA kernel to set a color that is used by 
       the GLSL shader 
    
    #. OpenGL draw calls use GL_POINT, but lines result via use of GLSL Geometry Shaders
       that amplify the input vertices using the momentum direction and polarization  

    Constituents
    ~~~~~~~~~~~~~
 
    #. kernel DAEPhotonsKernel
    #. shader DAEPhotonsShader

    """
    def __init__(self, dphotons, chroma ):
        """
        :param dphotons: DAEPhotons instance
        :param chroma: DAEChromaContext instance
        """
        self.dphotons = dphotons
        self.chroma = chroma
        self.interop = not self.chroma.dummy
        self.shader = DAEPhotonsShader(dphotons) 
        self.kernel = DAEPhotonsKernel(dphotons) if self.interop else None 
        self.propagator = DAEPhotonsPropagator(dphotons, chroma) if self.interop else None
        self.invalidate_buffers()
        pass
        self.create_buffer_count = 0 
        self.draw_count = 0 

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
        self.create_buffer_count += 1
        log.warn("############ create_buffer [count %s]  ##################### %s " % (self.create_buffer_count, repr(data.data.dtype)) )
        vbo = DAEVertexBuffer( data.data, data.indices, force_attribute_zero=data.force_attribute_zero, shader=self.shader  )
        self.interop_gl_to_cuda(vbo)
        return vbo

    def interop_gl_to_cuda(self, buf ):
        """ 
        Registering the VBO with CUDA, by creation of cuda_gl BufferObject 
        """
        if not self.interop:return
        buf.cuda_buffer_object = self.chroma.make_cuda_buffer_object(buf.vertices_id)

    def interop_cuda_to_gl(self, buf ):
        """
        Ends CUDA responsibility, allows OpenGL access for drawing 
        """
        if not self.interop:return
        buf.cuda_buffer_object.unregister() 

    def interop_call_cuda_kernel(self, buf):
        """
        Invoke CUDA kernel with VBO argument, 
        allowing VBO changes.
        """ 
        if not self.interop:return

        import pycuda.driver as cuda_driver
        buf_mapping = buf.cuda_buffer_object.map()
        vbo_dev_ptr = buf_mapping.device_ptr()

        #self.kernel( vbo_dev_ptr, self.dphotons.qcount )  ## huh why the dynamic qcount ? 
        self.propagator.propagate( vbo_dev_ptr, max_steps=1 )   

        cuda_driver.Context.synchronize()
        buf_mapping.unmap()

    def draw(self):
        """
        Drawing the pbuffer, after kernel call to potentially modify it

        #. qcount specified count of elements to draw

        Need way to control the speed of propagate steps, and a trigger.
        """
        self.draw_count += 1

        propagate = self.draw_count % 10 == 0    ## noddy slow down technique

        qcount = self.dphotons.qcount   

        gl.glPointSize(self.dphotons.param.fphopoint)  
       
        #self.kernel.update_constants()   # can polling param changes be avoided ?

        #if propagate:
        self.propagator.update_constants()   
        self.interop_call_cuda_kernel(self.pbuffer)

        self.interop_cuda_to_gl(self.pbuffer)

        self.pbuffer.draw(mode=gl.GL_POINTS,  what='', count=qcount,   offset=0, att=1 ) 

        self.interop_gl_to_cuda(self.pbuffer)

        gl.glPointSize(1)  


    def legacy_draw(self):
        """
        Drawing lbuffer (a doubled up pbuffer) with VertexAttrib offset tricks 
        (via att=1,2,3 corresponding to lines/startpoint/endpoint) to draw lines
        and points.

        In pre-history used both pbuffer and lbuffer to draw lines and points
        but that is no longer needed.
        """
        qcount = self.dphotons.qcount

        gl.glPointSize(self.dphotons.param.fphopoint)  

        self.lbuffer.draw(mode=gl.GL_LINES,   what='pc', count=2*qcount, offset=0, att=1 )

        self.lbuffer.draw(mode=gl.GL_POINTS,  what='pc', count=qcount,   offset=0, att=2 )    

        gl.glPointSize(1)  


if __name__ == '__main__':
    pass


