#!/usr/bin/env python
"""

* http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097?pgno=2
* http://andreask.cs.illinois.edu/PyCuda/Examples/SobelFilter
* http://andreask.cs.illinois.edu/PyCuda/Examples/GlInterop


"""
import ctypes
import numpy as np
import OpenGL.GL as gl
import OpenGL.raw.GL.VERSION.GL_1_1 as rawgl   #initially OpenGL.raw.GL as rawgl but only GL_1_1 has the glReadPixels symbol
import pycuda.gl as cuda_gl
import pycuda.driver as cuda_driver

from gpu.tools import get_cu_module, cuda_options, GPUFuncs, get_cu_source


class ImageProcessor(object):
    def __init__(self, w,h ):
        self.resize(w, h)
        self.source = PixelBuffer(w, h) 
        self.dest   = PixelBuffer(w, h, texture=True) 
        self.cuda_init()

    def resize(self, w,h ):
        self.image_width, self.image_height = w,h

    def process(self):
        self.source.load_from_framebuffer()
        self.cuda_process( self.source, self.dest )

    def cuda_init(self):
        raise Exception("sub classes expected to implement this")

    def cuda_process(self, source, dest):
        raise Exception("sub classes expected to implement this")

    def display(self):
        self.dest.draw()

    def cleanup(self):
        self.source.cleanup()
        self.dest.cleanup()



class Invert(ImageProcessor):
    def __init__(self, *args, **kwa):
        ImageProcessor.__init__(self, *args, **kwa)

    def cuda_init(self):
        invert = GPUFuncs(get_cu_module("invert.cu", options=cuda_options)).invert
        invert.prepare("PP")  # kernel takes two PBOs as arguments
        self.invert = invert

    def cuda_process(self, source, dest ):

        grid_dimensions   = (self.image_width//16,self.image_height//16)

        source_mapping = source.cuda_pbo.map()
        dest_mapping   = dest.cuda_pbo.map()

        source_ptr = source_mapping.device_ptr()
        dest_ptr = dest_mapping.device_ptr()

        self.invert.prepared_call(grid_dimensions, (16, 16, 1), source_ptr, dest_ptr )
        cuda_driver.Context.synchronize()

        source_mapping.unmap()
        dest_mapping.unmap()



class PixelBuffer(object):
    """
    * http://www.songho.ca/opengl/gl_pbo.html
    """
    def __init__(self, w, h, texture=False ):
        self.image_width = w
        self.image_height = h
        self.data = np.zeros((w*h,4),np.uint8)   # hmm can i just use NULL ?

        self.pbo = gl.glGenBuffers(1)
        target = gl.GL_PIXEL_UNPACK_BUFFER  # GL_ARRAY_BUFFER works too, **BUT 10X SLOWER**
        gl.glBindBuffer( target, self.pbo)
        gl.glBufferData( target, self.data, gl.GL_DYNAMIC_DRAW )

        buffer_size = gl.glGetBufferParameteriv( target, gl.GL_BUFFER_SIZE)
        assert buffer_size == w*h*4, (buffer_size, w*h*4 )

        gl.glBindBuffer( target, 0)

        self.cuda_pbo = cuda_gl.BufferObject(long(self.pbo))  # needs CUDA context

        self.tex = None
        if texture:
            self.tex = Texture(w, h)


    def load_from_framebuffer(self):
        """
        #. unregister tells cuda that OpenGL is accessing the PBO ?
        #. bind source.pbo as the PIXEL_PACK_BUFFER
        #. read pixels from framebuffer into PIXEL_PACK_BUFFER
        """
        assert self.cuda_pbo is not None

        self.cuda_pbo.unregister()

        gl.glBindBuffer( gl.GL_PIXEL_PACK_BUFFER , long(self.pbo)) 

        rawgl.glReadPixels(
             0,                  #start x
             0,                  #start y
             self.image_width,   #end   x
             self.image_height,  #end   y
             gl.GL_BGRA,            #format
             gl.GL_UNSIGNED_BYTE,   #output type
             ctypes.c_void_p(0))

        self.cuda_pbo = cuda_gl.BufferObject(long(self.pbo))  # have to re-make after unregister it seems

    def associate_to_tex(self, tex ):
        """
        Identifies this pbo as data source for the texture

        #. bind pbo.pbo as PIXEL_UNPACK_BUFFER 
        #. bind tex.tex as TEXTURE_2D
        #. associate the array to the texture

        From https://www.opengl.org/sdk/docs/man/docbook4/xhtml/glTexSubImage2D.xml

        If a non-zero named buffer object is bound to the
        GL_PIXEL_UNPACK_BUFFER target while a texture image is
        specified, the `data` parameter  is treated as a byte offset 
        into the buffer object's data store.

        """
        assert self.pbo is not None
        assert tex.tex is not None

        gl.glBindBuffer( gl.GL_PIXEL_UNPACK_BUFFER , long(self.pbo))
        gl.glBindTexture( gl.GL_TEXTURE_2D, tex.tex )

        target = gl.GL_TEXTURE_2D      # Specifies the target texture
        level = 0                   # level-of-detail number, Level 0 is the base image level
        xoffset = 0                 # Specifies a texel offset in the x direction within the texture array.
        yoffset = 0                 # Specifies a texel offset in the y direction within the texture array.
        width = self.image_width    # Specifies the width of the texture subimage.
        height = self.image_height  # Specifies the height of the texture subimage.
        format_ = gl.GL_BGRA           # Specifies the format of the pixel data.     BGRA is said to be best for performance
        type_ = gl.GL_UNSIGNED_BYTE    # Specifies the data type of the pixel data
        data = ctypes.c_void_p(0)   # Specifies a pointer to the image data in memory.

        rawgl.glTexSubImage2D(target, level, xoffset, yoffset, width, height, format_, type_, data )

    def unbind(self):
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER , 0)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER , 0)

    def draw(self):
        assert self.tex is not None 
        self.associate_to_tex( self.tex )
        self.tex.display() 
        self.unbind() 

    def cleanup(self):

        #target = gl.GL_ARRAY_BUFFER
        target = gl.GL_PIXEL_UNPACK_BUFFER 
        gl.glBindBuffer( target, long(self.pbo))
        gl.glDeleteBuffers(1, long(self.pbo));
        gl.glBindBuffer( target, 0)

        self.pbo = None 
        self.cuda_pbo = None 

        if self.tex is not None:
            self.tex.cleanup() 



class Texture(object):
    def __init__(self, w, h ):
        self.image_width = w
        self.image_height = h

        self.tex = gl.glGenTextures(1)

        gl.glBindTexture( gl.GL_TEXTURE_2D, self.tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, w, h, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)

    def display(self):
        """ render a screen sized quad """
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glTexEnvf(gl.GL_TEXTURE_ENV, gl.GL_TEXTURE_ENV_MODE, gl.GL_REPLACE)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPushMatrix()
        gl.glLoadIdentity()
        gl.glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
        gl.glMatrixMode( gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.glViewport(0, 0, self.image_width, self.image_height)
        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2f(0.0, 0.0)
        gl.glVertex3f(-1.0, -1.0, 0.5)
        gl.glTexCoord2f(1.0, 0.0)
        gl.glVertex3f(1.0, -1.0, 0.5)
        gl.glTexCoord2f(1.0, 1.0)
        gl.glVertex3f(1.0, 1.0, 0.5)
        gl.glTexCoord2f(0.0, 1.0)
        gl.glVertex3f(-1.0, 1.0, 0.5)
        gl.glEnd()
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPopMatrix()
        gl.glDisable(gl.GL_TEXTURE_2D)


    def cleanup(self):
        gl.glDeleteTextures(self.tex);
        self.tex = None



if __name__ == '__main__':
    pass

