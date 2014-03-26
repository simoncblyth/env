#!/usr/bin/env python
#
# From $(glumpy-dir)/demos/histogram.py
#
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# glumpy is an OpenGL framework for the fast visualization of numpy arrays.
# Copyright (C) 2009-2012  Nicolas P. Rougier. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY NICOLAS P. ROUGIER ''AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
# EVENT SHALL NICOLAS P. ROUGIER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# The views and conclusions contained in the software and documentation are
# those of the authors and should not be interpreted as representing official
# policies, either expressed or implied, of Nicolas P. Rougier.
# -----------------------------------------------------------------------------
'''
'''
import numpy as np
import OpenGL.GL as gl
from glumpy import figure, Trackball
from glumpy.graphics import VertexBuffer, Shader


def cube(center=(0.0,0.0,0.0), size=(0.5,0.5,1.0), color = (1.0,1.0,1.0)):
    """
    Returns vertices, fill indices and outlines indices for a cube with given
    center, size and color.
    """
    cx,cy,cz = np.array(center)
    sx,sy,sz = np.array(size)/2.0
    p = ( (cx+sx, cy+sy, cz+sz),
          (cx-sx, cy+sy, cz+sz),
          (cx-sx, cy-sy, cz+sz),
          (cx+sx, cy-sy, cz+sz),
          (cx+sx, cy-sy, cz-sz),
          (cx+sx, cy+sy, cz-sz),
          (cx-sx, cy+sy, cz-sz),
          (cx-sx, cy-sy, cz-sz) )
    n = ( ( 0, 0, 1), (1, 0, 0), ( 0, 1, 0),
          (-1, 0, 1), (0,-1, 0), ( 0, 0,-1) )
    c = np.resize(color, (8,3))
    vertices = np.array(
        [ (p[0],n[0],c[0]), (p[1],n[0],c[1]), (p[2],n[0],c[2]), (p[3],n[0],c[3]),
          (p[0],n[1],c[0]), (p[3],n[1],c[3]), (p[4],n[1],c[4]), (p[5],n[1],c[5]),
          (p[0],n[2],c[0]), (p[5],n[2],c[5]), (p[6],n[2],c[6]), (p[1],n[2],c[1]),
          (p[1],n[3],c[1]), (p[6],n[3],c[6]), (p[7],n[3],c[7]), (p[2],n[3],c[2]),
          (p[7],n[4],c[7]), (p[4],n[4],c[4]), (p[3],n[4],c[3]), (p[2],n[4],c[2]),
          (p[4],n[5],c[4]), (p[7],n[5],c[7]), (p[6],n[5],c[6]), (p[5],n[5],c[5]) ], 
        dtype = [('position','f4',3), ('normal','f4',3), ('color','f4',3)] )
    I = np.repeat(np.arange(0,24,4),6).reshape(6,6) + [0,1,2,0,2,3]
    fill = I.astype(np.uint32).ravel()
    I = np.repeat(np.arange(0,24,4),8).reshape(6,8) + [0,1,1,2,2,3,3,0]
    outline = I.astype(np.uint32).ravel()
    return vertices, fill, outline


def bars(Z):
    """
    Each bar has:
      - 24 vertices
      - 36 indices for fill shape
      - 48 indices for outline
    """
    nx,ny = Z.shape
    vertices = np.zeros((nx*ny,24), dtype = [('position','f4',3),
                                             ('normal','f4',3),
                                             ('color','f4',3)] )
    fill    = np.zeros((nx*ny,36)).astype(np.uint32)
    outline = np.zeros((nx*ny,48)).astype(np.uint32)
    v,f,o = cube( (-0.5,-0.5,0.0), ( 1./nx, 1./ny, 0))
    fill[:]    = (np.arange(0,nx*ny)*24).reshape(nx*ny,1) + f
    outline[:] = (np.arange(0,nx*ny)*24).reshape(nx*ny,1) + o
    X,Y = np.mgrid[0:nx,0:ny]
    X = np.repeat(X,24).reshape(nx*ny,24)
    Y = np.repeat(Y,24).reshape(nx*ny,24)
    vertices[...] = v
    vertices['position'][:,:,0] += X/float(nx)+.5/nx
    vertices['position'][:,:,1] += Y/float(ny)+.5/ny

    Z = np.repeat(Z,12).reshape(nx*ny,12)
    vertices['position'][:,[0,1,2,3,4,5,8,11,12,15,18,19],2] = Z-Z.min()+0.001

    return VertexBuffer(vertices,fill), VertexBuffer(vertices,outline)


def gaussian(shape=(25,25), width=0.15, center=0.0):
    if type(shape) in [float,int]:
        shape = (shape,)
    if type(width) in [float,int]:
        width = (width,)*len(shape)
    if type(center) in [float,int]:
        center = (center,)*len(shape)
    grid=[]
    for size in shape:
        grid.append (slice(0,size))
    C = np.mgrid[tuple(grid)]
    R = np.zeros(shape)
    for i,size in enumerate(shape):
        if shape[i] > 1:
            R += (((C[i]/float(size-1))*2 - 1 - center[i])/width[i])**2
    return np.exp(-R/2)


# -----------------------------------------------------------------------------
if __name__ == '__main__':


    fig = figure(size=(640,480))
    trackball = Trackball(65, 135, 1., 2.)
    Z = 0.25*gaussian((32,32), center = (.5,.0))
    fill, outline = bars(Z)
    t = 0

    def gl_modelview_matrix():
        return gl.glGetDoublev( gl.GL_MODELVIEW_MATRIX )

    def gl_projection_matrix():
        return gl.glGetDoublev( gl.GL_PROJECTION_MATRIX )

    @fig.event
    def on_init():
        gl.glEnable( gl.GL_BLEND )
        gl.glEnable(gl.GL_NORMALIZE)
        gl.glEnable( gl.GL_LINE_SMOOTH )
        gl.glEnable( gl.GL_COLOR_MATERIAL )
        gl.glColorMaterial ( gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE )
        gl.glBlendFunc (gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glEnable (gl.GL_LIGHTING)
        gl.glEnable (gl.GL_LIGHT0)
        gl.glEnable (gl.GL_LIGHT1)
        gl.glEnable (gl.GL_LIGHT2)

    @fig.event
    def on_mouse_drag(x, y, dx, dy, button):
        trackball.drag_to(x,y,dx,dy)
        fig.redraw()

    @fig.event
    def on_draw():
        fig.clear()
        trackball.push()

        #print gl_modelview_matrix()
        print gl_projection_matrix()

        gl.glLightfv (gl.GL_LIGHT0, gl.GL_POSITION,(-1.0,1.0, 1.0, 1.0))
        gl.glLightfv (gl.GL_LIGHT1, gl.GL_POSITION,(1.0, 1.0, 1.0, 1.0))
        gl.glLightfv (gl.GL_LIGHT2, gl.GL_POSITION,(0.0,-1.0, 1.0, 1.0))
        gl.glEnable (gl.GL_LIGHTING)
        
        gl.glEnable( gl.GL_POLYGON_OFFSET_FILL )
        gl.glPolygonOffset (1, 1)
        gl.glPolygonMode( gl.GL_FRONT_AND_BACK, gl.GL_FILL )
        gl.glColor( 1.0, 1.0, 1.0, 1.0 )
        fill.draw( gl.GL_TRIANGLES, 'pnc' )
        gl.glDisable( gl.GL_LIGHTING )
        gl.glDisable( gl.GL_POLYGON_OFFSET_FILL )

        gl.glPolygonMode( gl.GL_FRONT_AND_BACK, gl.GL_LINE )
        gl.glDepthMask( gl.GL_FALSE )
        gl.glColor( 0.0, 0.0, 0.0, 0.25 )
        outline.draw( gl.GL_LINES, 'p' )
        gl.glDepthMask( gl.GL_TRUE )
        
        trackball.pop()

    @fig.timer(60.0)
    def timer(dt):
        global t
        t = t+dt
        vertices = fill.vertices
        c,s = np.cos(t), np.sin(t)
        T = 0.25*gaussian(shape = Z.shape, center = (0.5*c,0.5*s))
        T = np.repeat(T,12).reshape(T.size,12)
        vertices['position'][:,[0,1,2,3,4,5,8,11,12,15,18,19],2] = T-T.min()+0.001
        fill.upload()
        outline.upload()
        fig.redraw()
        

    fig.show()
