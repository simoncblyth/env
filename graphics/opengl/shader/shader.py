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
import logging, pprint
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

lines_ = lambda _:["%-0.3d : %s " % (i+1, line) for i, line in enumerate(_.split("\n"))]

shadertypes = {
      gl.GL_VERTEX_SHADER:'vertex',
      gl.GL_FRAGMENT_SHADER:'fragment',
      gl.GL_GEOMETRY_SHADER:'geometry',
      }

class Shader(object):
    def __init__(self, **kwa ):

        log.debug("%s %s " % (self.__class__.__name__, pprint.pformat(kwa)))

        vertex = kwa.pop('vertex',None)
        fragment = kwa.pop('fragment',None)
        geometry = kwa.pop('geometry',None)

        self.input_type = kwa.pop('geometry_input_type', gl.GL_POINTS )
        self.output_type = kwa.pop('geometry_output_type', gl.GL_LINE_STRIP )
        self.vertices_out = kwa.pop('geometry_vertices_out', 200 )

        self.program = gl.glCreateProgram()
        self.uniforms = {}
        self.attribs = {}

        self._compile( vertex,   gl.GL_VERTEX_SHADER, kwa  )
        self._compile( fragment, gl.GL_FRAGMENT_SHADER, kwa  )
        self._compile( geometry, gl.GL_GEOMETRY_SHADER, kwa )

        self.linked = False

    def __str__(self):
        return "\n".join( 
           
                       ["#### vertex"]   +  lines_(self.vertex_source)  +
                       ["#### geometry input_type %s output_type %s vertices_out %s" % (self.input_type, self.output_type, self.vertices_out)] +  lines_(self.geometry_source) + 
                       ["#### fragment"] +  lines_(self.fragment_source)  
                        )

    def _setup_geometry_shader(self):        
        """
        At what juncture can these be changed ? 
        """
        log.info("_setup_geometry_shader")
        log.info("_setup_geometry_shader   input_type  %s " % self.input_type)
        log.info("_setup_geometry_shader  output_type  %s " % self.output_type)
        log.info("_setup_geometry_shader  vertices_out %s " % self.vertices_out )
        gsx.glProgramParameteriEXT(self.program, gsa.GL_GEOMETRY_INPUT_TYPE_ARB, self.input_type )
        gsx.glProgramParameteriEXT(self.program, gsa.GL_GEOMETRY_OUTPUT_TYPE_ARB, self.output_type )
        gsx.glProgramParameteriEXT(self.program, gsa.GL_GEOMETRY_VERTICES_OUT_ARB, self.vertices_out )
    
    def _compile(self, source_, shader_type, kwa={} ):
        
        typename = shadertypes[shader_type] 
        setattr( self, '%s_source' % typename,   "")
        if source_ is None:return

        source = source_ % kwa 
        setattr( self, '%s_source' % typename,  source)

        log.debug("_compile shader %s " % shader_type )

        shader = gl.glCreateShader(shader_type)
        gl.glShaderSource(shader, [source,])
        gl.glCompileShader(shader)
        status = gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS)

        if not status:
            print "\n".join(lines_(source))            
            raise Exception(
                    'Shader compilation error %s : ' % shader_type  + gl.glGetShaderInfoLog(shader))
        else:
            gl.glAttachShader(self.program, shader)

        if typename == 'geometry': 
            self._setup_geometry_shader()


    def link(self):
        """
        #. confirmed observation that when doing this link before every draw 
           without the already linked check, the photons flash 
           (appear for a single draw then disappear)

        #. but this seems not to happen 
           with the simple debug shader (which skips geometry shading)
           which makes less use of uniforms ?  CHECK THIS

        """
        if self.linked:
            #log.info("already linked, skip ")
            return

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

    def bind_attribute(self, index , name ):
        """
        #. must do this before linking the program
        #. make attrib accessible from shader
        #. this is equivalent to layout qualifier more recent GLSL 
  
           * http://www.opengl.org/wiki/Layout_Qualifier_(GLSL)

        * https://www.khronos.org/opengles/sdk/docs/man/xhtml/glBindAttribLocation.xml

        Attribute variable name-to-generic attribute index bindings for a
        program object can be explicitly assigned at any time by calling
        glBindAttribLocation. Attribute bindings do not go into effect until
        glLinkProgram is called. After a program object has been linked successfully,
        the index values for generic attributes remain fixed (and their values can be
        queried) until the next link command occurs.

        Applications are not allowed to bind any of the standard OpenGL vertex
        attributes using this command, as they are bound automatically when needed. Any
        attribute binding that occurs after the program object has been linked will not
        take effect until the next time the program object is linked.

        """
        gl.glBindAttribLocation( self.program, index, name )  

    def use(self):
        gl.glUseProgram(self.program)

    @classmethod
    def unuse(cls):
        """
        Unuse whichever program currently used
        """
        gl.glUseProgram(0)

    def uniformf(self, name, *vals):
        loc = self.uniforms.get(name, gl.glGetUniformLocation(self.program,name))
        self.uniforms[name] = loc
        log.info("uniformf [%s] %s %s " % (loc, name, repr(vals)))  

        if len(vals) in range(1, 5):
            { 1 : gl.glUniform1f,
              2 : gl.glUniform2f,
              3 : gl.glUniform3f,
              4 : gl.glUniform4f
            }[len(vals)](loc, *vals)

    def uniformi(self, name, *vals):
        loc = self.uniforms.get(name, gl.glGetUniformLocation(self.program,name))
        self.uniforms[name] = loc
        log.info("uniformi [%s] %s %s " % (loc, name, repr(vals)))  

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

