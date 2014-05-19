#!/usr/bin/env python
"""
Interop steps gleaned from raycaster PBO usage:

#. import pycuda.gl as cuda_gl                  # special GL enabled CUDA context  (handle in DAEChromaContext)
#. cuda_pbo = cuda_gl.BufferObject(long(pbo))   # needs CUDA context, pbo is OpenGL buffer id
#. cuda_pbo.unregister()                        # ends CUDA responsibility, allows OpenGL access for drawing 

#. pbo_mapping = cuda_pbo.map()                 # CUDA takes responsibility 
#. pbo_device_ptr = pbo_mapping.device_ptr()    # pointer to device memory to pass as kernel call argument 
#. pbo_mapping.unmap()                          # after the kernel call, declares end of CUDA usage

Currently are using the simple and inefficient technique 
of recreating the VBOs whenever need to change anything.

Ruminations

#. having separate VBOs for drawing lines and points is inconvenient, especially 
   when have CUDA in the mix

   * should be able to draw points from the lines VBO via indices control
     (ie have one vertex array and which is used with two different indices arrays)
    
   * how about flag control photon selections ? glumpy VertexBuffer glues together
     vertices and indices as pair of ctor arguments : they do not need to go together.

     ie could split that off and do the indices selection on CPU in similar manner
     to current ? But what about on GPU selection, it just needs bit field comparison to 
     uniform argument mask or history bitfield ... hmm but still would need to 
     pull out the selection bits CPU side to setup the OpenGL indices ? So little gain.
     Does OpenGL have some concept of a mask array ?

     Can control color in the kernel, so make unselected invisible (set alpha to 0)

     * :google:`OpenGL CUDA vertex selection`
     * http://www.opengl.org/wiki/Vertex_Rendering
 
   * 2nd point of line pair can be calulated in the kernel, with line length being a uniform
 
#. need to retain drawing of initial photon positions/directions for non-chroma installs 


Interop Refs

Using PrimitiveRestart

#. http://www.drdobbs.com/parallel/cuda-supercomputing-for-the-masses-part/225200412

   

"""

import ctypes
import OpenGL.GL as gl
import OpenGL.GLUT as glut
import glumpy as gp

class DAEVertexBuffer(gp.graphics.VertexBuffer):
    """
    Inherit from glumpy VertexBuffer in order to experiment with the DrawElements 
    call : attempting for partial draws.
    """
    def __init__(self, *args, **kwa ):
        interop = kwa.pop('interop',False) 
        gp.graphics.VertexBuffer.__init__(self, *args )

        #if interop:
        #   import pycuda.gl as cuda_gl   # unclear where to do this
        #   self.vertices_cuda = cuda_gl.BufferObject(long(self.vertices_id))
        #


    def draw( self, mode=gl.GL_QUADS, what='pnctesf', offset=0, count=None ):
        """ 
        :param mode: primitive to draw
        :param what: attribute multiple choice by first letter
        :param offset: integer element array buffer offset, default 0
        :param count: number of elements, default None corresponds to all in self.indices

        Buffer offset default of 0 corresponds to glumpy original None, (ie (void*)0 )
        the integet value is converted with `ctypes.c_void_p(offset)`   
        allowing partial buffer drawing.

        * http://pyopengl.sourceforge.net/documentation/manual-3.0/glDrawElements.html
        * http://stackoverflow.com/questions/11132716/how-to-specify-buffer-offset-with-pyopengl
        * http://pyopengl.sourceforge.net/documentation/pydoc/OpenGL.arrays.vbo.html


        ====================  ==============
        type
        ====================  ==============
        GL_UNSIGNED_BYTE        0:255
        GL_UNSIGNED_SHORT,      0:65535
        GL_UNSIGNED_INT         0:4.295B
        ====================  ==============

        ===================   ====================================
           mode 
        ===================   ====================================
          GL_POINTS
          GL_LINE_STRIP
          GL_LINE_LOOP
          GL_LINES
          GL_TRIANGLE_STRIP
          GL_TRIANGLE_FAN
          GL_TRIANGLES
          GL_QUAD_STRIP
          GL_QUADS
          GL_POLYGON
        ===================   ====================================


        The what letters, 'pnctesf' define the meaning of the arrays via 
        enabling appropriate attributes.

        ==================  ==================   ================   =====
        gl***Pointer          GL_***_ARRAY          Att names         *
        ==================  ==================   ================   =====
         Color                COLOR                color              c
         EdgeFlag             EDGE_FLAG            edge_flag          e
         FogCoord             FOG_COORD            fog_coord          f
         Normal               NORMAL               normal             n
         SecondaryColor       SECONDARY_COLOR      secondary_color    s
         TexCoord             TEXTURE_COORD        tex_coord          t 
         Vertex               VERTEX               position           p
         VertexAttrib         N/A             
        ==================  ==================   ================   =====


        """
        if count is None:
           count = self.indices.size   # this is what the glumpy original does
        pass

        gl.glPushClientAttrib( gl.GL_CLIENT_VERTEX_ARRAY_BIT )
        gl.glBindBuffer( gl.GL_ARRAY_BUFFER, self.vertices_id )
        gl.glBindBuffer( gl.GL_ELEMENT_ARRAY_BUFFER, self.indices_id )

        for attribute in self.generic_attributes:
            attribute.enable()
        for c in self.attributes.keys():
            if c in what:
                self.attributes[c].enable()

        gl.glDrawElements( mode, count, gl.GL_UNSIGNED_INT, ctypes.c_void_p(offset) )

        gl.glBindBuffer( gl.GL_ELEMENT_ARRAY_BUFFER, 0 ) 
        gl.glBindBuffer( gl.GL_ARRAY_BUFFER, 0 ) 
        gl.glPopClientAttrib( )



if __name__ == '__main__':
    pass




