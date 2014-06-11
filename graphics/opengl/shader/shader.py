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

        self.input_type = kwa.pop('geometry_input_type', gl.GL_POINTS )
        self.output_type = kwa.pop('geometry_output_type', gl.GL_LINE_STRIP )
        self.vertices_out = kwa.pop('geometry_vertices_out', 200 )

        self.program = gl.glCreateProgram()
        self.uniforms = {}
        self.attribs = {}


        if not vertex is None:
            vertex_source = vertex % kwa 
            self._compile( vertex_source , gl.GL_VERTEX_SHADER )
        else:
            vertex_source = ""
        pass
        self.vertex_source = vertex_source

        if not fragment is None:
            fragment_source = fragment % kwa 
            self._compile( fragment_source , gl.GL_FRAGMENT_SHADER )
        else:
            fragment_source = ""
        pass
        self.fragment_source = fragment_source

        if not geometry is None:
            assert gsa and gsx
            geometry_source = geometry % kwa 
            self._compile( geometry_source , gl.GL_GEOMETRY_SHADER )
            self._setup_geometry_shader()
        else:
            geometry_source = ""
        pass
        self.geometry_source = geometry_source

    def __str__(self):
        source_ = lambda _:["%2s : %s " % (i, line) for i, line in enumerate(_.split("\n"))]
        return "\n".join( 
                       ["#### vertex"]   +  source_(self.vertex_source)  +
                       ["#### geometry"] +  source_(self.geometry_source) + 
                       ["#### fragment"] +  source_(self.fragment_source)  
                        )

    def _setup_geometry_shader(self):        
        """
        At what juncture can these be changed ? 
        """
        gsx.glProgramParameteriEXT(self.program, gsa.GL_GEOMETRY_INPUT_TYPE_ARB, self.input_type )
        gsx.glProgramParameteriEXT(self.program, gsa.GL_GEOMETRY_OUTPUT_TYPE_ARB, self.output_type )
        gsx.glProgramParameteriEXT(self.program, gsa.GL_GEOMETRY_VERTICES_OUT_ARB, self.vertices_out )
    
    def _compile(self, source, shader_type):
        if source is None:return

        log.debug("_compile shader %s " % shader_type )

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
        log.debug("link ")
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


    def attrib(self, name ):
        if not name in self.attribs:
            loc = self.attribs.get(name, gl.glGetAttribLocation( self.program, name ))
            self.attribs[name] = loc 
        return self.attribs[name] 



if __name__ == '__main__':
    pass 

