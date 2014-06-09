#!/usr/bin/python
"""

**DO NOT EDIT THIS : INSTEAD USE geometry_shader_structured.py**


http://stackoverflow.com/questions/3936368/geometry-shader-doesnt-do-anything-when-fed-gl-points

Comment `glutMainLoop()` so can check for compile errors reveals::

    ERROR: 0:8: Attempt to redeclare 'distance' as a variable
    ERROR: 0:17: Attempt to use 'distance' as a variable
    ERROR: 0:18: Attempt to use 'distance' as a variable
    ERROR: 0:25: Attempt to use 'distance' as a variable
    ERROR: 0:26: Attempt to use 'distance' as a variable

    ERROR: One or more attached shaders not successfully compiled


After renaming `distance` to `mydistance` succeeds to compile and run 
with or without USE_POINTS=True.

::

    File "errorchecker.pyx", line 50, in OpenGL_accelerate.errorchecker._ErrorChecker.glCheckError (src/errorchecker.c:854)
    OpenGL.error.GLError: GLError(
        err = 1282,
        description = 'invalid operation',
        baseOperation = glUseProgram,
        cArguments = (1L,)


"""
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.GL.ARB.geometry_shader4 import *
from OpenGL.GL.EXT.geometry_shader4 import *

import numpy
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

    glUseProgram(shader)
    glUniform1f(glGetUniformLocation(shader, "mydistance"), rot/10000.0)

    # difference #1
    glBegin(GL_POINTS if USE_POINTS else GL_LINES)
    for x in [-2.5, 0, 2.5]:
        for y in [-2.5, 0, 2.5]:
            glVertexAttrib1f(7, random.uniform(0.0, 1.0))
            glVertexAttrib3f(0, x, y, 0)
            # difference #2
            if not USE_POINTS:
                glVertexAttrib1f(7, random.uniform(0.0, 1.0))
                glVertexAttrib3f(0, x, y, 0)
    glEnd()
    glUseProgram(0)

    glutSwapBuffers()

def key(*args):
    if args[0] == '\x1b':
        sys.exit(0);

def reshape(width, height):
    aspect = float(width)/float(height) if (height>0) else 1.0
    glViewport(0, 0, width, height)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    gluPerspective(45.0,
                   aspect,
                   1.0, 100.0)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glutPostRedisplay()




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

shader = glCreateProgram()
vertex_shader = glCreateShader(GL_VERTEX_SHADER)
geometry_shader = glCreateShader(GL_GEOMETRY_SHADER)
fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)

# difference #3
glProgramParameteriEXT(shader, GL_GEOMETRY_INPUT_TYPE_ARB, GL_POINTS if USE_POINTS else GL_LINES)
glProgramParameteriEXT(shader, GL_GEOMETRY_OUTPUT_TYPE_ARB, GL_LINE_STRIP)
glProgramParameteriEXT(shader, GL_GEOMETRY_VERTICES_OUT_ARB, 200)

glAttachShader(shader, vertex_shader)
glAttachShader(shader, geometry_shader)
glAttachShader(shader, fragment_shader)

glShaderSource(vertex_shader, """
attribute float color;
varying float geom_color;
void main(void) {
  gl_Position = gl_Vertex;
  geom_color = color;
}
""")
glCompileShader(vertex_shader)
print glGetShaderInfoLog(vertex_shader)

glShaderSource(geometry_shader, """
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
""")
glCompileShader(geometry_shader)
print glGetShaderInfoLog(geometry_shader)

glShaderSource(fragment_shader, """
varying float frag_color;
void main(void) {
  gl_FragColor = vec4(frag_color,1.0-frag_color,frag_color,1);
}
""")
glCompileShader(fragment_shader)
print glGetShaderInfoLog(fragment_shader)

glLinkProgram(shader)
print glGetProgramInfoLog(shader)

glBindAttribLocation(shader, 7, "color")

glLinkProgram(shader)
print glGetProgramInfoLog(shader)


glutMainLoop()
