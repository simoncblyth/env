#!/usr/bin/env python

import OpenGL.GL as gl
import pycuda.gl as cuda_gl

class PixelBuffer(object):
    def __init__(self, data ):
        self.pbo = gl.glGenBuffers(1)
        gl.glBindBuffer( gl.GL_ARRAY_BUFFER, self.pbo)
        gl.glBufferData( gl.GL_ARRAY_BUFFER, data, gl.GL_DYNAMIC_DRAW)
        gl.glBindBuffer( gl.GL_ARRAY_BUFFER, 0)
        self.cuda_pbo = None
        self.make_cuda_pbo()

    def make_cuda_pbo(self):  
        """After have CUDA context alive"""
        self.cuda_pbo = cuda_gl.BufferObject(long(self.pbo))

    def cleanup(self):
        gl.glBindBuffer( gl.GL_ARRAY_BUFFER, long(self.pbo))
        gl.glDeleteBuffers(1, long(self.pbo));
        gl.glBindBuffer( gl.GL_ARRAY_BUFFER, 0)

        self.pbo = None 
        self.cuda_pbo = None 


class Texture(object):
    def __init__(self, w, h ):
        self.tex = gl.glGenTextures(1)
        gl.glBindTexture( gl.GL_TEXTURE_2D, self.tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, w, h, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)

    def cleanup(self):
        gl.glDeleteTextures(self.tex);
        self.tex = None



if __name__ == '__main__':
    pass

