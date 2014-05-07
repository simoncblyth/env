#!/usr/bin/env python

import sys
import numpy as np
import OpenGL
import OpenGL.GL as gl
import OpenGL.GLU as glu
import OpenGL.GLUT as glut

from glumpy.graphics import VertexBuffer


def on_display_0():
    global cube, theta, phi, frame, time, timebase

    frame += 1
    time = glut.glutGet( glut.GLUT_ELAPSED_TIME )
    if (time - timebase > 1000):
        print frame*1000.0/(time-timebase)
        timebase = time;		
        frame = 0;

    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    gl.glPushMatrix()
    gl.glRotatef(theta, 0,0,1)
    gl.glRotatef(phi, 0,1,0)

    gl.glDisable( gl.GL_BLEND )
    gl.glEnable( gl.GL_LIGHTING )
    gl.glEnable( gl.GL_DEPTH_TEST )
    gl.glEnable( gl.GL_POLYGON_OFFSET_FILL )

    gl.glPolygonMode( gl.GL_FRONT_AND_BACK, gl.GL_FILL )
    cube.draw( gl.GL_QUADS, 'pnc' )

    gl.glDisable( gl.GL_POLYGON_OFFSET_FILL )
    gl.glEnable( gl.GL_BLEND )
    gl.glDisable( gl.GL_LIGHTING )
    gl.glPolygonMode( gl.GL_FRONT_AND_BACK, gl.GL_LINE )
    gl.glDepthMask( gl.GL_FALSE )
    gl.glColor( 0.0, 0.0, 0.0, 0.5 )

    cube.draw( gl.GL_QUADS, 'p' )

    gl.glDepthMask( gl.GL_TRUE )
    gl.glPopMatrix()

    glut.glutSwapBuffers()


def on_display_1():
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    gl.glPushMatrix()
    gl.glRotatef(theta, 0,0,1)
    gl.glRotatef(phi, 0,1,0)

    gl.glDisable( gl.GL_LIGHTING )
    gl.glEnable( gl.GL_BLEND )
    gl.glPolygonMode( gl.GL_FRONT_AND_BACK, gl.GL_LINE )
    gl.glDepthMask( gl.GL_FALSE )
    gl.glColor( 0.0, 0.0, 0.0, 0.5 )

    cube.draw( gl.GL_QUADS, 'pnc' )

    gl.glPopMatrix()
    glut.glutSwapBuffers()


on_display = on_display_0

    
def on_reshape(width, height):
    gl.glViewport(0, 0, width, height)
    gl.glMatrixMode( gl.GL_PROJECTION )
    gl.glLoadIdentity( )
    glu.gluPerspective( 45.0, float(width)/float(height), 2.0, 10.0 )
    gl.glMatrixMode( gl.GL_MODELVIEW )
    gl.glLoadIdentity( )
    gl.glTranslatef( 0.0, 0.0, -5.0 )

def on_keyboard(key, x, y):
    if key == '\033':
        sys.exit()

def on_timer(value):
    global theta, phi
    theta += 0.25
    phi += 0.25
    glut.glutPostRedisplay()
    glut.glutTimerFunc(10, on_timer, 0)

def on_idle():
    global theta, phi
    theta += 0.25
    phi += 0.25
    glut.glutPostRedisplay()


if __name__ == '__main__':
    p = ( ( 1, 1, 1), (-1, 1, 1), (-1,-1, 1), ( 1,-1, 1),
          ( 1,-1,-1), ( 1, 1,-1), (-1, 1,-1), (-1,-1,-1) )
    n = ( ( 0, 0, 1), (1, 0, 0), ( 0, 1, 0),
          (-1, 0, 1), (0,-1, 0), ( 0, 0,-1) );
    c = ( ( 1, 1, 1), ( 1, 1, 0), ( 1, 0, 1), ( 0, 1, 1),
          ( 1, 0, 0), ( 0, 0, 1), ( 0, 1, 0), ( 0, 0, 0) );

    vertices = np.array(
        [ (p[0],n[0],c[0]), (p[1],n[0],c[1]), (p[2],n[0],c[2]), (p[3],n[0],c[3]),
          (p[0],n[1],c[0]), (p[3],n[1],c[3]), (p[4],n[1],c[4]), (p[5],n[1],c[5]),
          (p[0],n[2],c[0]), (p[5],n[2],c[5]), (p[6],n[2],c[6]), (p[1],n[2],c[1]),
          (p[1],n[3],c[1]), (p[6],n[3],c[6]), (p[7],n[3],c[7]), (p[2],n[3],c[2]),
          (p[7],n[4],c[7]), (p[4],n[4],c[4]), (p[3],n[4],c[3]), (p[2],n[4],c[2]),
          (p[4],n[5],c[4]), (p[7],n[5],c[7]), (p[6],n[5],c[6]), (p[5],n[5],c[5]) ], 
        dtype = [('position','f4',3), ('normal','f4',3), ('color','f4',3)] )


    glut.glutInit(sys.argv)
    glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGB | glut.GLUT_DEPTH)
    glut.glutCreateWindow("Python VBO")
    glut.glutReshapeWindow(400, 400)
    glut.glutDisplayFunc(on_display)
    glut.glutReshapeFunc(on_reshape)
    glut.glutKeyboardFunc(on_keyboard)
    glut.glutTimerFunc(10, on_timer, 0)
    #glut.glutIdleFunc(on_idle)

    gl.glPolygonOffset( 1, 1 )
    gl.glClearColor(1,1,1,1);
    gl.glEnable( gl.GL_DEPTH_TEST )
    gl.glEnable( gl.GL_COLOR_MATERIAL )
    gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)
    gl.glBlendFunc( gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA )
    gl.glEnable( gl.GL_LIGHT0 )
    gl.glLight( gl.GL_LIGHT0, gl.GL_DIFFUSE,  (1.0,1.0,1.0,1.0) )
    gl.glLight( gl.GL_LIGHT0, gl.GL_AMBIENT,  (0.1,0.1,0.1,1.0) )
    gl.glLight( gl.GL_LIGHT0, gl.GL_SPECULAR, (0.0,0.0,0.0,1.0) )
    gl.glLight( gl.GL_LIGHT0, gl.GL_POSITION, (0.0,1.0,2.0,1.0) )
    gl.glEnable( gl.GL_LINE_SMOOTH )

    theta, phi = 0, 0
    frame, time, timebase = 0, 0, 0
    cube = VertexBuffer(vertices)

    glut.glutMainLoop()

