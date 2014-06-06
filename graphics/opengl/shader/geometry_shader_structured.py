#!/usr/bin/python
"""
http://stackoverflow.com/questions/3936368/geometry-shader-doesnt-do-anything-when-fed-gl-points

"""
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from OpenGL.GL.ARB.geometry_shader4 import *
from OpenGL.GL.EXT.geometry_shader4 import *

from shader import Shader

import numpy
import numpy.linalg as linalg
import random
from math import sin, cos

shader = None

USE_POINTS = True
#USE_POINTS = False

def update(*args):
    glutTimerFunc(33, update, 0)
    glutPostRedisplay()

def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    t = glutGet(GLUT_ELAPSED_TIME)

    rot = t % (10 * 1000)
    theta = 2 * 3.141592 * (rot / 10000.0)

    glLoadIdentity()
    gluLookAt(-10*sin(theta), -10*cos(theta),   0,
                0,   0,   0,
                0,   0,   1)

    shader.bind()
    shader.uniformf( "mydistance" , rot/10000.0)

    glBegin(geometry_input_type)
    for x in [-2.5, 0, 2.5]:
        for y in [-2.5, 0, 2.5]:
            glVertexAttrib1f(7, random.uniform(0.0, 1.0))
            glVertexAttrib3f(0, x, y, 0)
            if not USE_POINTS:
                glVertexAttrib1f(7, random.uniform(0.0, 1.0))
                glVertexAttrib3f(0, x, y, 0)
    glEnd()

    shader.unbind()

    glutSwapBuffers()

def key(*args):
    if args[0] == '\x1b':
        sys.exit(0);

def reshape(width, height):
    aspect = float(width)/float(height) if (height>0) else 1.0
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, aspect, 1.0, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glutPostRedisplay()


vertex = """
    attribute float color;
    varying float geom_color;
    void main(void) {
      gl_Position = gl_Vertex;
      geom_color = color;
    }
    """
    
geometry_input_type = GL_POINTS if USE_POINTS else GL_LINES

def setup_geometry_shader(program, input_type=GL_POINTS, output_type=GL_LINE_STRIP, vertices_out=200 ): 
    glProgramParameteriEXT(program, GL_GEOMETRY_INPUT_TYPE_ARB, input_type)
    glProgramParameteriEXT(program, GL_GEOMETRY_OUTPUT_TYPE_ARB, output_type)
    glProgramParameteriEXT(program, GL_GEOMETRY_VERTICES_OUT_ARB, vertices_out)

geometry = """
    #version 120
    #extension GL_EXT_geometry_shader4 : enable

    varying in float geom_color[1];
    varying out float frag_color;

    uniform float mydistance;

    void main(void)
    {
     int x, y;

     for(x=-1; x<=1; x+=1) {
       for(y=-1; y<=1; y+=1) {
         gl_Position = gl_PositionIn[0];
         gl_Position.x += x * mydistance;
         gl_Position.y += y * mydistance;
         gl_Position.z -= 2.0;
         gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
         frag_color = geom_color[0];
         EmitVertex();

         gl_Position = gl_PositionIn[0];
         gl_Position.x += x * mydistance;
         gl_Position.y += y * mydistance;
         gl_Position.z += 2.0;
         gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
         frag_color = geom_color[0];
         EmitVertex();
         EndPrimitive();
       }
     }
    }
    """

fragment = """
    varying float frag_color;
    void main(void) {
      gl_FragColor = vec4(frag_color,1.0-frag_color,frag_color,1);
    }
    """



if __name__ == '__main__':

    glutInit([])
    glutInitDisplayString("rgba>=8 depth>16 double")
    glutInitWindowSize(1280, 720)
    glutCreateWindow("Geometry Shader")

    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(key)

    glutTimerFunc(33, update, 0)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_POINT_SMOOTH)
    glEnable(GL_LINE_SMOOTH)

    shader = Shader( vertex, fragment, geometry )
    setup_geometry_shader( shader.program, geometry_input_type )
    shader._link()

    glBindAttribLocation(shader.program, 7, "color")

    glutMainLoop()

