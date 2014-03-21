#!/usr/bin/env python
"""
http://stackoverflow.com/questions/13179565/how-to-get-vbos-to-work-with-python-and-pyopengl

pygame is nicely terse compared to GLUT setup
"""
import numpy as np
import pygame
from OpenGL.GL import *
from ctypes import *

width, height = 800,600
nvert = 10000  # a million takes a second or so to display
vertices = np.random.random((nvert,3)).flatten() - 0.5


pygame.init ()
screen = pygame.display.set_mode ((width, height), pygame.OPENGL|pygame.DOUBLEBUF, 24)

glViewport (0, 0, width, height)
glClearColor (0.5, 0.5, 0.5, 1.0)


#GL_VERTEX_ARRAY,  
#If enabled, the vertex array is enabled for writing and used during rendering
#when glArrayElement, glDrawArrays, glDrawElements, glDrawRangeElements
#glMultiDrawArrays, or glMultiDrawElements is called. See glVertexPointer.
#
#  http://stackoverflow.com/questions/11806823/glenableclientstate-deprecated
#

capability = GL_VERTEX_ARRAY   # 
glEnableClientState(capability)

n = 1   # number of buffer object names to generate
vbo = glGenBuffers(n)    # name of the buffer, an integer

target = GL_ARRAY_BUFFER
buffer_ = vbo

glBindBuffer(target, buffer_)


target = GL_ARRAY_BUFFER    #  target buffer objec
size = len(vertices)*4       # size in bytes of the buffer object's new data store.
data = (c_float*len(vertices))(*vertices) # pointer to data that will be copied into the data store for initialization, or NULL if no data is to be copied.
usage = GL_STATIC_DRAW    # expected usage pattern   GL_(STREAM/STATIC/DYNAMIC)_(DRAW/READ/COPY)   how used/modified

glBufferData(target, size, data, usage )


running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    glClear(GL_COLOR_BUFFER_BIT)


    size, type_, stride, pointer = 3, GL_FLOAT, 0, None
    glVertexPointer(size, type_, stride, pointer)  # 4th param **MUST BE None, not 0**

    first, count = 0, nvert
    glDrawArrays(GL_TRIANGLES, first, count)

    pygame.display.flip ()


