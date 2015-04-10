#!/usr/bin/env python
"""

* http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097?pgno=2
* http://andreask.cs.illinois.edu/PyCuda/Examples/SobelFilter
* http://andreask.cs.illinois.edu/PyCuda/Examples/GlInterop


"""
import numpy as np
import ctypes
import OpenGL.GL as gl
import OpenGL.raw.GL.VERSION.GL_1_1 as rawgl   #initially OpenGL.raw.GL as rawgl but only GL_1_1 has the glReadPixels symbol
import pycuda.gl as cuda_gl


class PixelBuffer(object):
    """
    * http://www.songho.ca/opengl/gl_pbo.html
    """
    def __init__(self, size, texture=False ):
        self.texture = texture
        self.make_pbo(size)
        self.tex = None
        if self.texture:
            self.tex = Texture(size)

    size = property(lambda self:(self.image_width, self.image_height))
    npixels = property(lambda self:self.image_width*self.image_height)

    def resize(self, size): 
        if self.size == size:return
        self.cleanup()
        self.make_pbo( size )
        if self.texture:
            self.tex.resize(size)

    def make_pbo(self, size):
        w, h = size
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


    def load_from_framebuffer(self, fx=0., fy=0., fw=1., fh=1.):
        """
        #. unregister tells cuda that OpenGL is accessing the PBO ?
        #. bind source.pbo as the PIXEL_PACK_BUFFER
        #. read pixels from framebuffer into PIXEL_PACK_BUFFER

        Using ordinary pyopengl rather than rawgl might be possible, 
        also juggling multiple PBOs may allow this to be faster, as OpenGL
        has to wait for rendering to finish before reading.

        * http://pyopengl.sourceforge.net/documentation/manual-3.0/glReadPixels.html
        * https://www.opengl.org/discussion_boards/showthread.php/165780-PBO-glReadPixels-not-so-fast

        """
        assert self.cuda_pbo is not None

        self.cuda_pbo.unregister()

        gl.glBindBuffer( gl.GL_PIXEL_PACK_BUFFER , long(self.pbo)) 

        x,y = int(fx*self.image_width),int(fy*self.image_height)
        w,h = int(fw*self.image_width),int(fh*self.image_height)

        rawgl.glReadPixels(
             x,                     # start x
             y,                     # start y
             w,                     # end   x
             h,                     # end   y
             gl.GL_BGRA,            # format
             gl.GL_UNSIGNED_BYTE,   # output type
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
        level = 0                      # level-of-detail number, Level 0 is the base image level
        xoffset = 0                    # Specifies a texel offset in the x direction within the texture array.
        yoffset = 0                    # Specifies a texel offset in the y direction within the texture array.
        width = self.image_width       # Specifies the width of the texture subimage.
        height = self.image_height     # Specifies the height of the texture subimage.
        format_ = gl.GL_BGRA           # Specifies the format of the pixel data.     BGRA is said to be best for performance
        type_ = gl.GL_UNSIGNED_BYTE    # Specifies the data type of the pixel data
        data = ctypes.c_void_p(0)      # Specifies a pointer to the image data in memory.

        rawgl.glTexSubImage2D(target, level, xoffset, yoffset, width, height, format_, type_, data )

    def unbind(self):
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER , 0)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER , 0)

    def draw(self,*args,**kwa):
        assert self.tex is not None 
        self.associate_to_tex( self.tex )
        self.tex.display(*args,**kwa) 
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
    def __init__(self, size ):
        self.tex = self.make_tex(size)

    def make_tex(self, size):
        w, h = size
        self.image_width = w
        self.image_height = h

        tex = gl.glGenTextures(1)

        gl.glBindTexture( gl.GL_TEXTURE_2D, tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, w, h, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)

        return tex

    size = property(lambda self:(self.image_width, self.image_height))

    def resize(self, size): 
        if self.size == size:return
        self.cleanup()
        self.tex = self.make_tex( size )

    def display(self, fx=0., fy=0., fw=1., fh=1.):
        """ 
        Parameters specify lower left corner (x,y) and (width,height) on the screen to draw the texture, 
        in units of the screen width and height

        :param fx: fraction of window 
        :param fy:
        :param fw:
        :param fh:

        Hmm how to do this in modern OpenGL ?

        * need buffers to pass to shader 

          * vertices and texcoords of one GL_QUADS


        * http://gamedev.stackexchange.com/questions/35486/map-and-fill-texture-using-pbo-opengl-3-3

        """ 
        x, y = int(fx*self.image_width), int(fy*self.image_height)
        width, height = int(fw*self.image_width), int(fh*self.image_height)  

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
        gl.glViewport(x,y, width, height)
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

        # otherwise get grey silhouettes 
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_LIGHTING)


    def cleanup(self):
        gl.glDeleteTextures(self.tex);
        self.tex = None



