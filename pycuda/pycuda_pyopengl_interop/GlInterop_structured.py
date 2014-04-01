#!/usr/bin/env python
"""
Draws a rotating teapot, using cuda to invert the RGB value each frame

http://wiki.tiker.net/PyCuda/Examples

Based on GL interoperability example by Peter Berrington.
From /usr/local/env/chroma_env/build/build_pycuda/pycuda/examples/wiki-examples

"""
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL.ARB.vertex_buffer_object import *
from OpenGL.GL.ARB.pixel_buffer_object import *
import OpenGL.raw.GL.VERSION.GL_1_1 as rawgl   #initially OpenGL.raw.GL as rawgl but only GL_1_1 has the glReadPixels symbol

import numpy, sys, time
import pycuda.driver as cuda_driver
import pycuda.gl as cuda_gl
import pycuda.gl.autoinit

from pycuda.compiler import SourceModule


initial_size = 512,512
current_size = initial_size
window = None     # Number of the glut window.

time_of_last_draw = 0.0
time_of_last_titleupdate = 0.0
frames_per_second = 0.0
frame_counter = 0
heading,pitch,bank = [0.0]*3


from interop_pixel_buffer import PixelBuffer, Texture

glinterop = None



class GlInterop(object):
    def __init__(self, w,h ):
        """
        #. create source and destination pixel buffer objects for processing
        #. create texture for blitting to screen
        """
        self.resize((w, h))
        self.data = numpy.zeros((w*h,4),numpy.uint8)
        self.source = PixelBuffer(self.data) 
        self.dest   = PixelBuffer(self.data) 
        self.output = Texture( w,h ) 
        self.animate = True
        self.enable_cuda = True

    def resize(self, size ):
        self.image_width, self.image_height = size
    def toggle_animate(self):
        self.animate = not self.animate 
    def toggle_enable_cuda(self):
        self.enable_cuda = not self.enable_cuda 

    def init_cuda(self):
        """
        #. "PP" indicates that the invert function will take two PBOs as arguments
        """
        cuda_module = SourceModule("""
            __global__ void invert(unsigned char *source, unsigned char *dest)
            {
              int block_num        = blockIdx.x + blockIdx.y * gridDim.x;
              int thread_num       = threadIdx.y * blockDim.x + threadIdx.x;
              int threads_in_block = blockDim.x * blockDim.y;
              //Since the image is RGBA we multiply the index 4.
              //We'll only use the first 3 (RGB) channels though
              int idx              = 4 * (threads_in_block * block_num + thread_num);
              dest[idx  ] = 255 - source[idx  ];
              dest[idx+1] = 255 - source[idx+1];
              dest[idx+2] = 255 - source[idx+2];
            }
            """)
        invert = cuda_module.get_function("invert")
        invert.prepare("PP")   

        self.invert = invert


    def cuda_process(self, source, dest ):
        """ Use PyCuda """

        grid_dimensions   = (self.image_width//16,self.image_height//16)

        source_mapping = source.cuda_pbo.map()
        dest_mapping   = dest.cuda_pbo.map()

        self.invert.prepared_call(grid_dimensions, (16, 16, 1),
            source_mapping.device_ptr(),
              dest_mapping.device_ptr())

        cuda_driver.Context.synchronize()

        source_mapping.unmap()
        dest_mapping.unmap()


    def process_image(self):

        self.read_into_pbo( self.source )

        self.cuda_process( self.source, self.dest )

        self.associate_pbo_to_tex( self.dest , self.output )


    def read_into_pbo(self, pbo ):
        """
        #. unregister tells cuda that OpenGL is accessing the PBO ?
        #. tell OpenGL that source.pbo is now the PIXEL_PACK_BUFFER
        #. read pixels from OpenGL framebuffer into the PIXEL_PACK_BUFFER, ie the source.pbo
        """
        assert pbo.pbo is not None
        assert pbo.cuda_pbo is not None
        pbo.cuda_pbo.unregister()

        glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, long(pbo.pbo)) 

        rawgl.glReadPixels(
             0,                  #start x
             0,                  #start y
             self.image_width,   #end   x
             self.image_height,  #end   y
             GL_BGRA,            #format
             GL_UNSIGNED_BYTE,   #output type
             ctypes.c_void_p(0))

        pbo.cuda_pbo = cuda_gl.BufferObject(long(pbo.pbo))  # have to re-make after unregister it seems

    def associate_pbo_to_tex(self, pbo, tex ):
        """
        Identifes dest.pbo as the data source for the output.tex 

        #. bind dest.pbo as PIXEL_UNPACK_BUFFER 
        #. bind output.tex as TEXTURE_2D
        #. associate the array to the texture

        From https://www.opengl.org/sdk/docs/man/docbook4/xhtml/glTexSubImage2D.xml

        If a non-zero named buffer object is bound to the
        GL_PIXEL_UNPACK_BUFFER target while a texture image is
        specified, the `data` parameter  is treated as a byte offset 
        into the buffer object's data store.

        """
        assert pbo.pbo is not None
        assert tex.tex is not None

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, long(pbo.pbo))
        glBindTexture(GL_TEXTURE_2D, tex.tex )

        target = GL_TEXTURE_2D      # Specifies the target texture
        level = 0                   # level-of-detail number, Level 0 is the base image level
        xoffset = 0                 # Specifies a texel offset in the x direction within the texture array.
        yoffset = 0                 # Specifies a texel offset in the y direction within the texture array.
        width = self.image_width    # Specifies the width of the texture subimage.
        height = self.image_height  # Specifies the height of the texture subimage.
        format_ = GL_BGRA           # Specifies the format of the pixel data.     BGRA is said to be best for performance
        type_ = GL_UNSIGNED_BYTE    # Specifies the data type of the pixel data
        data = ctypes.c_void_p(0)   # Specifies a pointer to the image data in memory.

        rawgl.glTexSubImage2D(target, level, xoffset, yoffset, width, height, format_, type_, data )


    def display_image(self):
        """ render a screen sized quad """

        width, height = self.image_width, self.image_height

        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glEnable(GL_TEXTURE_2D)
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
        glMatrixMode( GL_MODELVIEW)
        glLoadIdentity()
        glViewport(0, 0, width, height)
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0)
        glVertex3f(-1.0, -1.0, 0.5)
        glTexCoord2f(1.0, 0.0)
        glVertex3f(1.0, -1.0, 0.5)
        glTexCoord2f(1.0, 1.0)
        glVertex3f(1.0, 1.0, 0.5)
        glTexCoord2f(0.0, 1.0)
        glVertex3f(-1.0, 1.0, 0.5)
        glEnd()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glDisable(GL_TEXTURE_2D)

        # hmm these seem out of place 
        glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0)


    def cleanup(self):
        self.output.cleanup()
        self.source.cleanup()
        self.dest.cleanup()




def init_gl():
    Width, Height = current_size
    glClearColor(0.1, 0.1, 0.5, 1.0)
    glDisable(GL_DEPTH_TEST)
    glViewport(0, 0, Width, Height)
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, Width/float(Height), 0.1, 10.0)
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    glEnable(GL_LIGHT0)
    red   = ( 1.0, 0.1, 0.1, 1.0 )
    white = ( 1.0, 1.0, 1.0, 1.0 )
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE,  red  )
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, white)
    glMaterialf( GL_FRONT_AND_BACK, GL_SHININESS, 60.0)

def resize(Width, Height):
    global current_size
    current_size = Width, Height

    glinterop.resize(current_size)

    glViewport(0, 0, Width, Height)        # Reset The Current Viewport And Perspective Transformation
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60.0, Width/float(Height), 0.1, 10.0)

def do_tick():
    global time_of_last_titleupdate, frame_counter, frames_per_second
    if ((time.clock () * 1000.0) - time_of_last_titleupdate >= 1000.):
        frames_per_second = frame_counter                   # Save The FPS
        frame_counter = 0  # Reset The FPS Counter
        szTitle = "%d FPS" % (frames_per_second )
        glutSetWindowTitle ( szTitle )
        time_of_last_titleupdate = time.clock () * 1000.0
    frame_counter += 1

# The function called whenever a key is pressed. Note the use of Python tuples to pass in: (key, x, y)
def keyPressed(*args):
    global glinterop
 
    # If escape is pressed, kill everything.
    if args[0] == '\033':
        print 'Closing..'
        glinterop.cleanup()
        exit()
    elif args[0] == 'a':
        print 'toggling animation'
        glinterop.toggle_animate()
    elif args[0] == 'e':
        print 'toggling cuda'
        glinterop.toggle_enable_cuda()

def idle():
    global heading, pitch, bank
    if glinterop.animate:
        heading += 0.2
        pitch   += 0.6
        bank    += 1.0

    glutPostRedisplay()

def display():
    global glinterop
    try:
        render_scene()
        if glinterop.enable_cuda:
            glinterop.process_image()
            glinterop.display_image()
        glutSwapBuffers()
    except:
        from traceback import print_exc
        print_exc()
        from os import _exit
        _exit(0)


def render_scene():
    glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)# Clear Screen And Depth Buffer
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity ()      # Reset The Modelview Matrix
    glTranslatef(0.0, 0.0, -3.0);
    glRotatef(heading, 1.0, 0.0, 0.0)
    glRotatef(pitch  , 0.0, 1.0, 0.0)
    glRotatef(bank   , 0.0, 0.0, 1.0)
    glViewport(0, 0, current_size[0],current_size[1])
    glEnable(GL_LIGHTING)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)
    glutSolidTeapot(1.0)
    do_tick()#just for fps display..
    return True




def main():
    global window
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH)
    glutInitWindowSize(*initial_size)
    glutInitWindowPosition(0, 0)
    window = glutCreateWindow("PyCuda GL Interop Example")
    glutDisplayFunc(display)
    glutIdleFunc(idle)
    glutReshapeFunc(resize)
    glutKeyboardFunc(keyPressed)
    glutSpecialFunc(keyPressed)
    init_gl()

    global glinterop
    glinterop = GlInterop(*initial_size)
    glinterop.init_cuda()

    glutMainLoop()

# Print message to console, and kick off the main to get it rolling.
if __name__ == "__main__":
    print "Hit ESC key to quit, 'a' to toggle animation, and 'e' to toggle cuda"
    main()
