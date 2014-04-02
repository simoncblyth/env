#!/usr/bin/env python
"""
Draws a rotating teapot, using cuda to invert the RGB value each frame

http://wiki.tiker.net/PyCuda/Examples

Based on GL interoperability example by Peter Berrington.
From /usr/local/env/chroma_env/build/build_pycuda/pycuda/examples/wiki-examples

NB due to use of PIXEL_UNPACK_BUFFER rather than ARRAY_BUFFER this is 10x faster 
than the original example


"""

import logging
log = logging.getLogger(__name__)

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

#from OpenGL.GL.ARB.vertex_buffer_object import *
#from OpenGL.GL.ARB.pixel_buffer_object import *
#import OpenGL.raw.GL.VERSION.GL_1_1 as rawgl   #initially OpenGL.raw.GL as rawgl but only GL_1_1 has the glReadPixels symbol

import numpy, sys, time
import pycuda.gl.autoinit
from env.pycuda.pycuda_pyopengl_interop import Invert, Generate


initial_size = 512,512
current_size = initial_size
window = None     # Number of the glut window.

time_of_last_draw = 0.0
time_of_last_titleupdate = 0.0
frames_per_second = 0.0
frame_counter = 0
heading,pitch,bank = [0.0]*3


glinterop = None

class GlInterop(object):
    def __init__(self, processor):
        self.processor = processor
        self.animate = True
        self.enable_cuda = True

    def toggle_animate(self):
        self.animate = not self.animate 
    def toggle_enable_cuda(self):
        self.enable_cuda = not self.enable_cuda 

    def process(self,*args,**kwa):
        self.processor.process(*args,**kwa)
    def display(self,*args,**kwa):
        self.processor.display(*args,**kwa)
    def cleanup(self):
        self.processor.cleanup()

    def resize(self, w,h):
        print "resize not implemented"


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

    glinterop.resize(*current_size)

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
            fx,fy,fw,fh = 0.,0.,1.,1.
            #fx,fy,fw,fh = 0.25,0.25,0.5,0.5  # attempt to filter just mid portion not giving what expect get multiple teapots in the filter region
            glinterop.process(fx=fx,fy=fy,fw=fw,fh=fh)
            glinterop.display(fx=fx,fy=fy,fw=fw,fh=fh)
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
    w,h = initial_size
    processor = Invert(w,h)
    #processor = Generate(*initial_size)
    glinterop = GlInterop(processor)

    glutMainLoop()

# Print message to console, and kick off the main to get it rolling.
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print "Hit ESC key to quit, 'a' to toggle animation, and 'e' to toggle cuda"
    main()
