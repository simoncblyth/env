#!/usr/bin/python
"""

http://stackoverflow.com/questions/3936368/geometry-shader-doesnt-do-anything-when-fed-gl-points

"""
import sys, logging
log = logging.getLogger(__name__)

import OpenGL.GL as gl
import OpenGL.GLU as glu
import OpenGL.GLUT as glut

from shader import Shader
import numpy, random, math

shader = None

def update(*args):
    glut.glutTimerFunc(33, update, 0)
    glut.glutPostRedisplay()

def input_geometry_points(color_attrib):
    """
    Is there something special about vertex attribute 0 ? implicitly the vertex position

    * http://www.khronos.org/webgl/wiki/WebGL_and_OpenGL_Differences#Vertex_Attribute_0
    * http://stackoverflow.com/questions/13348885/why-does-opengl-drawing-fail-when-vertex-attrib-array-zero-is-disabled

    On desktop GL, vertex attribute 0 has special semantics. 
    First, it must be enabled as an array, or no geometry will be drawn.
     
    """
    position_attrib = 0   # murky history of OpenGL immediate mode explains this

    gl.glBegin(gl.GL_POINTS)
    for x in [-1.5, 0, 2.5]:
        for y in [-1.5, 0, 2.5]:
            gl.glVertexAttrib1f(color_attrib, random.uniform(0.0, 1.0))
            gl.glVertexAttrib3f(position_attrib, x, y, 0)
        pass
    gl.glEnd()


def input_geometry_lines(color_attrib):
    position_attrib = 0   # murky history of OpenGL immediate mode explains this

    gl.glBegin(gl.GL_LINE_STRIP)
    for x in [-1.5, 0, 2.5]:
        for y in [-1.5, 0, 2.5]:
            gl.glVertexAttrib1f(color_attrib, random.uniform(0.0, 1.0))
            gl.glVertexAttrib3f(position_attrib, x, y, 0)

            gl.glVertexAttrib1f(color_attrib, random.uniform(0.0, 1.0))
            gl.glVertexAttrib3f(position_attrib, x, y, 5)

            gl.glVertexAttrib1f(color_attrib, random.uniform(0.0, 1.0))
            gl.glVertexAttrib3f(position_attrib, x, y, 10)

        pass
    gl.glEnd()



    




def display():
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    t = glut.glutGet(glut.GLUT_ELAPSED_TIME)

    rot = t % (10 * 1000)
    theta = 2 * 3.141592 * (rot / 10000.0)

    gl.glLoadIdentity()
    glu.gluLookAt(-10*math.sin(theta), -10*math.cos(theta),   0,
                    0,   0,   0,
                    0,   0,   1)


    if shader.enable: 
        shader.use()   
        shader.uniformf( "mydistance" , rot/10000.0)
        shader.uniformi( "iparam" ,     1, 1, 1, 1 )
    
    #input_geometry_points(color_attrib = shader.attrib('color'))
    input_geometry_lines(color_attrib = shader.attrib('color'))

    if shader.enable:
        shader.unuse()  # glUseProgram(0)


    glut.glutSwapBuffers()


def key(*args):
    if args[0] == '\x1b':
        sys.exit(0);

def reshape(width, height):
    aspect = float(width)/float(height) if (height>0) else 1.0
    gl.glViewport(0, 0, width, height)
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    glu.gluPerspective(45.0, aspect, 1.0, 100.0)
    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()
    glut.glutPostRedisplay()

    



vertex = """
#version 120
attribute float color;

varying float geom_color;
void main(void) {
   gl_Position = gl_Vertex;
   geom_color = color;
}
"""

geometry = """
#version 120
#extension GL_EXT_geometry_shader4 : enable

varying in float geom_color[1];
varying out float frag_color;

uniform ivec4 iparam; 
uniform float mydistance;

void main(void)
{
 int x, y;

 for(x=-iparam.x ; x<=iparam.x; x+=1) {
   for(y=-iparam.y; y<=iparam.y; y+=1) {
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
#version 120
varying float frag_color;
void main(void) {
   gl_FragColor = vec4(frag_color,1.0-frag_color,frag_color,1);
}
"""



if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    glut.glutInit([])
    glut.glutInitDisplayString("rgba>=8 depth>16 double")
    glut.glutInitWindowSize(1280, 720)
    glut.glutCreateWindow("Geometry Shader")

    glut.glutDisplayFunc(display)
    glut.glutReshapeFunc(reshape)
    glut.glutKeyboardFunc(key)

    glut.glutTimerFunc(33, update, 0)

    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glEnable(gl.GL_POINT_SMOOTH)
    gl.glEnable(gl.GL_LINE_SMOOTH)


    cfg = {}
    cfg['vertex'] = vertex
    cfg['fragment'] = fragment
    cfg['geometry'] = geometry 

    #cfg['geometry_input_type'] = gl.GL_POINTS 
    cfg['geometry_input_type'] = gl.GL_LINES

    cfg['geometry_output_type'] = gl.GL_LINE_STRIP 
    cfg['geometry_vertices_out'] = 200

    shader = Shader(**cfg)
    shader.link()

    shader.enable = False


    glut.glutMainLoop()

