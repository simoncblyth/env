#!/usr/bin/env python
"""
Usage. Compile and link vertex, fragment and geometry shaders prior to the main loop::

    vertex = """ """ 
    geometry = """ """ 
    fragment = """ """ 
    shader = Shader( vertex, fragment, geometry )  # shader source parameters  
    shader.link()

Drawing code probably needs to set uniforms and attribs, do this
inbetween a `shader.bind` and `shader.unbind` pair eg::

    shader.bind()   # glUseProgram

    shader.uniformf( "mydistance" , rot/10000.0)
    input_geometry(color_attrib = shader.attrib('color'))

    shader.unbind()  # glUseProgram(0)

"""
import logging
log = logging.getLogger(__name__)

import ctypes
import OpenGL.GL as gl

try:
    import OpenGL.GL.ARB.geometry_shader4 as gsa
except ImportError:
    gsa = None 

try:
    import OpenGL.GL.EXT.geometry_shader4 as gsx
except ImportError:
    gsx = None 


class Shader(object):
    def __init__(self, vertex=None, fragment=None, geometry=None, **kwa ):

        self.program = gl.glCreateProgram()
        self.uniforms = {}
        self.attribs = {}

        if not vertex is None:
            self._compile( vertex , gl.GL_VERTEX_SHADER )

        if not fragment is None:
            self._compile( fragment , gl.GL_FRAGMENT_SHADER )

        if not geometry is None:
            assert gsa and gsx
            self._compile( geometry , gl.GL_GEOMETRY_SHADER )
            self._setup_geometry_shader( **kwa )

    def _setup_geometry_shader(self, **kwa):        
        input_type = kwa.pop('geometry_input_type', gl.GL_POINTS )
        output_type = kwa.pop('geometry_output_type', gl.GL_LINE_STRIP )
        vertices_out = kwa.pop('geometry_vertices_out', 200 )

        gsx.glProgramParameteriEXT(self.program, gsa.GL_GEOMETRY_INPUT_TYPE_ARB, input_type )
        gsx.glProgramParameteriEXT(self.program, gsa.GL_GEOMETRY_OUTPUT_TYPE_ARB, output_type )
        gsx.glProgramParameteriEXT(self.program, gsa.GL_GEOMETRY_VERTICES_OUT_ARB, vertices_out )
    
    def _compile(self, source, shader_type):
        if source is None:return

        log.info("_compile shader %s " % shader_type )

        shader = gl.glCreateShader(shader_type)
        gl.glShaderSource(shader, [source,])
        gl.glCompileShader(shader)
        status = gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS)

        if not status:
            raise Exception(
                    'Shader compilation error %s : ' % shader_type  + gl.glGetShaderInfoLog(shader))
        else:
            gl.glAttachShader(self.program, shader)

    def link(self):
        log.info("link ")
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
        log.info("bind ")
        gl.glUseProgram(self.program)

    @classmethod
    def unbind(cls):
        """
        Unbinds whichever program currently used
        """
        log.info("unbind ")
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


    def attrib(self, name ):
        if not name in self.attribs:
            loc = self.attribs.get(name, gl.glGetAttribLocation( self.program, name ))
            self.attribs[name] = loc 
        return self.attribs[name] 



if __name__ == '__main__':
    pass 

