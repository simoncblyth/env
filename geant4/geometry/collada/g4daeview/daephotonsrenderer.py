#!/usr/bin/env python

import logging, pprint
log = logging.getLogger(__name__)

import OpenGL.GL as gl

from daevertexbuffer import DAEVertexBuffer
from daephotonsshader import DAEPhotonsShader
from daephotonspresenter import DAEPhotonsPresenter


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
 
    #. shader DAEPhotonsShader
    #. presenter DAEPhotonsPresenter

    """
    def __init__(self, drawable, chroma, config ):
        """
        :param drawable: instance of DAEDrawable subclass eg DAEPhotons  
        :param chroma: DAEChromaContext instance
        """
        self.drawable = drawable
        self.config = config
        self.chroma = chroma
        self.max_slots = self.config.args.max_slots

        self.interop = not chroma.dummy
        log.debug("%s.__init__" % self.__class__.__name__ )
        self.shader = DAEPhotonsShader(drawable) 
        self.presenter = DAEPhotonsPresenter(drawable, chroma, debug=int(self.config.args.debugkernel)) if self.interop else None
        self.invalidate_buffers()
        pass
        self.create_buffer_count = 0 
        self.draw_count = 0 


    def update_constants(self): 
        self.presenter.update_constants()

    def _get_shaderkey(self):
        return self.shader.shaderkey
    def _set_shaderkey(self, shaderkey):
        self.shader.shaderkey = shaderkey 
    shaderkey = property(_get_shaderkey, _set_shaderkey)



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
           self._pbuffer = self.create_buffer(self.drawable.array)  
        return self._pbuffer
    pbuffer = property(_get_pbuffer, doc="point DAEVertexBuffer, without doubling : used with geometry shader to generate 2nd vertices and lines ")  

    def _get_lbuffer(self):
        assert 0, "in use ? "
        if self._lbuffer is None:
           self._lbuffer = self.create_buffer(self.drawable.array)  
        return self._lbuffer
    lbuffer = property(_get_lbuffer, doc="line DAEVertexBuffer, with doubled vertices : not used when geometry shader available")  

    def create_buffer(self, _array ):
        """
        :param data: DAEPhotonsData instance
        :return: DAEVertexBuffer instance

        #. buffer creation does not belong in DAEPhotonsData as OpenGL specific
        """
        if _array.vbodata is None:return None
        self.create_buffer_count += 1

        log.info("_array %s " % repr(_array)) 

        log.debug("############ create_buffer [count %s]  ##################### %s " % (self.create_buffer_count, repr(_array.vbodata.dtype)) )
        vbo = DAEVertexBuffer( self, _array.vbodata, _array.indices, max_slots=_array.max_slots, force_attribute_zero=_array.force_attribute_zero )

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

    def draw(self, mode=gl.GL_POINTS, slot=0):
        """
        Drawing the pbuffer, after kernel call to potentially modify it

        #. qcount specified count of elements to draw

        """
        self.draw_count += 1

        qcount = self.drawable.qcount   

        #log.info("%s draw slot %s draw_count %s qcount %s " % (self.__class__.__name__, slot, self.draw_count, qcount ))

        gl.glPointSize(self.drawable.param.fphopoint)  
       
        if self.drawable.animate:
            self.presenter.interop_present(self.pbuffer, max_slots=self.max_slots)

        self.interop_cuda_to_gl(self.pbuffer)

        self.pbuffer.draw(mode,  what='', count=qcount,   offset=0, att=1, slot=slot ) 

        self.interop_gl_to_cuda(self.pbuffer)

        gl.glPointSize(1)  


    def multidraw(self, mode=gl.GL_LINE_STRIP, slot=None, counts=None, firsts=None, drawcount=None, extrakey=None):
        """

        #. prior to MultiDraw the presenter kernel is invoked, allowing per-draw VBO changes (eg used for 
           animation time interpolation changing VBO top slot)

        """
        self.draw_count += 1

        qcount = self.drawable.qcount   

        #log.info("%s multidraw slot %s draw_count %s qcount %s " % (self.__class__.__name__, slot, self.draw_count, qcount ))

        gl.glPointSize(self.drawable.param.fphopoint)  
       
        if self.drawable.animate:
            self.presenter.interop_present(self.pbuffer, max_slots=self.max_slots)

        self.interop_cuda_to_gl(self.pbuffer)


        if not extrakey is None:
            shaderkey = self.shaderkey 
            if not shaderkey == extrakey:        
                self.shaderkey = extrakey
                self.pbuffer.multidraw(mode,  what='', drawcount=qcount, slot=slot, counts=counts, firsts=firsts) 
                self.shaderkey = shaderkey
            pass
        pass

        self.pbuffer.multidraw(mode,  what='', drawcount=qcount, slot=slot, counts=counts, firsts=firsts) 


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
        qcount = self.drawable.qcount

        gl.glPointSize(self.drawable.param.fphopoint)  

        self.lbuffer.draw(mode=gl.GL_LINES,   what='pc', count=2*qcount, offset=0, att=1 )

        self.lbuffer.draw(mode=gl.GL_POINTS,  what='pc', count=qcount,   offset=0, att=2 )    

        gl.glPointSize(1)  


if __name__ == '__main__':
    pass


