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
    def __init__(self, **cfg ):
        self._program = None
        self._cfg = None
        self.cfg = cfg   # setter invalidates

    def invalidate(self):
        """
        No _program as cleanup is required for that
        """
        log.info("invalidate")
        self.uniforms = {}
        self.attribs = {}
    
    def _get_program(self):
        if self._program is None:
            self.create() 
        return self._program
    program = property(_get_program)

    def _get_cfg(self):
        return self._cfg
    def _set_cfg(self, cfg ):
        """
        Change in config causes deletion of program and shader, 
        On next use they are recreated.
        """
        log.info("_set_cfg ")
        if cfg == self._cfg:
            log.info("_set_cfg unchanged %s " % repr(cfg))
            return
        self._cfg = cfg
        self.config_changed()
    cfg = property(_get_cfg,_set_cfg)

    def config_changed(self):
        log.info("config_changed")
        self.delete()

    input_type = property(lambda self:self.cfg.get('geometry_input_type', gl.GL_POINTS ))
    output_type = property(lambda self:self.cfg.get('geometry_output_type', gl.GL_LINE_STRIP ))
    vertices_out = property(lambda self:self.cfg.get('geometry_vertices_out', 200 ))

    vertex_source = property(lambda self:self.cfg.get('vertex',""))
    fragment_source = property(lambda self:self.cfg.get('fragment',""))
    geometry_source = property(lambda self:self.cfg.get('geometry',""))

    shaderkey = property(lambda self:self.cfg.get('shaderkey',"no-shaderkey ?"))

    def __str__(self):
        return "%s:%s" % (self.__class__.__name__, self.shaderkey)



    def create(self): 
        cfg = self.cfg 
        self._program = gl.glCreateProgram()
        log.info("create _program %s " % self._program )
        self.linked = False
        self.uniforms = {}
        self.attribs = {}

        _attached_shaders = []
        vertex   = self.make_shader( self.vertex_source,   gl.GL_VERTEX_SHADER )
        fragment = self.make_shader( self.fragment_source, gl.GL_FRAGMENT_SHADER )
        geometry = self.make_shader( self.geometry_source, gl.GL_GEOMETRY_SHADER )

        for shader in filter(None,[vertex, fragment, geometry]):      
            gl.glAttachShader(self._program, shader)
            _attached_shaders.append(shader)
        pass

        log.info("_attached_shaders %s " % repr(_attached_shaders)) 
        self._attached_shaders = _attached_shaders

        if not geometry is None:
            self._setup_geometry_shader()
        pass

    def _setup_geometry_shader(self):        
        """
        At what juncture can these be changed ? 
        """
        log.info("_setup_geometry_shader")
        log.info("_setup_geometry_shader   input_type  %s " % self.input_type)
        log.info("_setup_geometry_shader  output_type  %s " % self.output_type)
        log.info("_setup_geometry_shader  vertices_out %s " % self.vertices_out )
        gsx.glProgramParameteriEXT(self._program, gsa.GL_GEOMETRY_INPUT_TYPE_ARB, self.input_type )
        gsx.glProgramParameteriEXT(self._program, gsa.GL_GEOMETRY_OUTPUT_TYPE_ARB, self.output_type )
        gsx.glProgramParameteriEXT(self._program, gsa.GL_GEOMETRY_VERTICES_OUT_ARB, self.vertices_out )

 
    def delete(self):
        """
        https://www.khronos.org/opengles/sdk/docs/man/xhtml/glDetachShader.xml
        """
        log.info("delete")
        if self._program is None:return

        if hasattr(self, '_attached_shaders'):
            for shader in self._attached_shaders:
                gl.glDetachShader( self._program, shader ) 
                gl.glDeleteShader( shader )
            pass
        pass
        gl.glDeleteProgram(self._program)
        self._program = None

    def source(self):
        gsmry = "#### geometry input/output/vertices " + "/".join(map(str,filter(None,(self.input_type, self.output_type, self.vertices_out,))))
        return "\n".join( 
                       ["#### vertex"]   +  lines_(self.vertex_source)  +
                       [gsmry]           +  lines_(self.geometry_source) + 
                       ["#### fragment"] +  lines_(self.fragment_source)  
                        )
   
    def make_shader(self, source , shader_type):
        if source is None or len(source) == 0:
            return None
        log.debug("make_shader %s " % shader_type )

        shader = gl.glCreateShader(shader_type)
        gl.glShaderSource(shader, [source,])
        gl.glCompileShader(shader)
        status = gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS)

        if not status:
            print "\n".join(lines_(source))            
            raise Exception(
                    'Shader compilation error %s : ' % shader_type  + gl.glGetShaderInfoLog(shader))

        return shader




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

