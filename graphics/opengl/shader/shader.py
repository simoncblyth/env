#!/usr/bin/env python
"""
"""
import logging
log = logging.getLogger(__name__)

import ctypes
import OpenGL.GL as gl

class Shader(object):
    def __init__(self, vertex=None, fragment=None, geometry=None ):

        self.program = gl.glCreateProgram()
        self.uniforms = {}

        if not vertex is None:
            self._build_shader( vertex , gl.GL_VERTEX_SHADER )

        if not fragment is None:
            self._build_shader( fragment , gl.GL_FRAGMENT_SHADER )

        if not geometry is None:
            self._build_shader( geometry , gl.GL_GEOMETRY_SHADER )
    
    def _build_shader(self, source, shader_type):
        if source is None:return

        shader = gl.glCreateShader(shader_type)
        gl.glShaderSource(shader, [source,])
        gl.glCompileShader(shader)
        status = gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS)

        if not status:
            raise Exception(
                    'Shader compilation error %s : ' % shader_type  + gl.glGetShaderInfoLog(shader))
        else:
            gl.glAttachShader(self.program, shader)

    def _link(self):
        gl.glLinkProgram(self.program)
        ok = ctypes.c_int(0)
        gl.glGetProgramiv(self.program, gl.GL_LINK_STATUS, ctypes.byref(ok))

        if not ok:
            gl.glGetProgramiv(self.program, gl.GL_INFO_LOG_LENGTH, ctypes.byref(ok))
            infolog = gl.glGetProgramInfoLog(self.program) 
            raise Exception("Linking error "+ infolog )
        pass 
        self.linked = True

    def bind(self):
        gl.glUseProgram(self.program)

    @classmethod
    def unbind(cls):
        """
        Unbinds whichever program currently used
        """
        gl.glUseProgram(0)

    def uniformf(self, name, *vals):
        loc = self.uniforms.get(name, gl.glGetUniformLocation(self.program,name))
        self.uniforms[name] = loc

        if len(vals) in range(1, 5):
            { 1 : gl.glUniform1f,
              2 : gl.glUniform2f,
              3 : gl.glUniform3f,
              4 : gl.glUniform4f
            }[len(vals)](loc, *vals)

    def uniformi(self, name, *vals):
        loc = self.uniforms.get(name, gl.glGetUniformLocation(self.program,name))
        self.uniforms[name] = loc

        if len(vals) in range(1, 5):
            { 1 : gl.glUniform1i,
              2 : gl.glUniform2i,
              3 : gl.glUniform3i,
              4 : gl.glUniform4i
            }[len(vals)](loc, *vals)

    def uniformMatrix4f(self, name, mat):
        loc = self.uniforms.get(name, gl.glGetUniformLocation(self.program,name))
        self.uniforms[name] = loc
        gl.glUniformMatrix4fv(loc, 1, False, (ctypes.c_float * 16)(*mat))




if __name__ == '__main__':
    pass 

