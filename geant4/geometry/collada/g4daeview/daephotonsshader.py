#!/usr/bin/env python
"""

* http://www.cs.unh.edu/~cs770/docs/glsl-1.20-quickref.pdf

When using position_name="position" DAEVertexBuffer does
the traditional glVertexPointer setup that furnishes gl_Vertex
to the shader. 

Legacy way prior to move to generic attributes::

    gl_Position = vec4( gl_Vertex.xyz , 1.) ; 
    //vMomdir = vec4( 100.,100.,100., 1.) ;

"""
from env.graphics.opengl.shader.shader import Shader
import logging
log = logging.getLogger(__name__)

from env.graphics.color.wav2RGB import wav2RGB_glsl

import OpenGL.GL as gl


vertex = r"""#version 120
// vertex : that simply passes through to geometry shader 

uniform vec4  fparam; 
uniform ivec4 iparam; 

attribute vec4 position_weight;
attribute vec4 direction_wavelength;
attribute vec4 polarization_time;

//attribute uvec4 flags;
//attribute ivec4 last_hit_triangle;

varying vec4 vMomdir;
varying vec4 vColor ;

%(funcs)s

void main()
{
    gl_Position = vec4( position_weight.xyz, 1.) ; 
    vMomdir = fparam.x*vec4( direction_wavelength.xyz, 1.) ;
    vColor = wav2color( direction_wavelength.w );
    //vColor = vec4( 1.0, 0., 0., 1.);
}

""" % { 'funcs':wav2RGB_glsl }

vertex_debug = r"""#version 120
// vertex_debug : for use without geometry shader 

uniform vec4  fparam; 
uniform ivec4 iparam; 

attribute vec4 position_weight;
attribute vec4 direction_wavelength;

varying vec4 fColor;

%(funcs)s

void main()
{
    gl_Position = vec4( position_weight.xyz, 1.) ; 
    gl_Position.xyz += fparam.x*direction_wavelength.xyz ; 
    gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
    fColor = wav2color( direction_wavelength.w );
}

""" % { 'funcs':wav2RGB_glsl }

geometry = r"""#version 120
#extension GL_EXT_geometry_shader4 : enable

//  http://www.opengl.org/wiki/Geometry_Shader_Examples

varying in vec4 vMomdir[] ; 
varying in vec4 vColor[] ; 
varying out vec4 fColor ;

uniform  vec4 fparam; 
uniform ivec4 iparam; 

void main()
{

    gl_Position = gl_PositionIn[0];
    gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
    fColor = vColor[0] ;
    EmitVertex();

    gl_Position = gl_PositionIn[0];
    gl_Position.xyz += vMomdir[0].xyz ;
    gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
    fColor = vColor[0] ;
    EmitVertex();

    EndPrimitive();
}
"""

fragment = r"""

varying vec4 fColor ;

void main()
{
    gl_FragColor = fColor ;
    //gl_FragColor = vec4(.0, 1.0, 0, 1);
}
"""

class DAEPhotonsShader(object):
    """
    NEXT:
    #. Try to reconfig the shader to do lines and points rather than having two shaders
    """
    def __init__(self, dphotons ):

        log.info("%s" % (self.__class__.__name__))
        for k in (gl.GL_VERSION, gl.GL_SHADING_LANGUAGE_VERSION, gl.GL_EXTENSIONS ):
            v = gl.glGetString(k)  
            log.info("%s %s " % (k, "\n".join(v.split())  ))

        if dphotons.param.debugshader:
            log.debug("compiling debug shader")
            shader = Shader( vertex_debug , fragment, None )
        else:
            log.debug("compiling normal shader")
            #shader = Shader( vertex, fragment, geometry, geometry_output_type=gl.GL_POINTS )
            shader = Shader( vertex, fragment, geometry, geometry_output_type=gl.GL_LINE_STRIP )
        pass
        self.shader = shader
        self.dphotons = dphotons

        self._iparam = None
        self._fparam = None

    def init_uniforms(self):
        self.iparam = self.dphotons.param.shader_iparam
        self.fparam = self.dphotons.param.shader_fparam


    def _get_iparam(self):
        return self._iparam
    def _set_iparam(self, iparam):
        """
        """
        if iparam == self._iparam:
            return 
        self._iparam = iparam
        self.shader.uniformi("iparam", *self._iparam)
    iparam = property(_get_iparam, _set_iparam, doc="shader iparam uniform ")


    def _get_fparam(self):
        return self._fparam
    def _set_fparam(self, fparam):
        """
        """
        if fparam == self._fparam:
            return 
        self._fparam = fparam
        self.shader.uniformf("fparam", *self._fparam)
    fparam = property(_get_fparam, _set_fparam, doc="shader fparam uniform ")





    def __str__(self):
        return "%s\n%s " % (self.__class__.__name__, str(self.shader))


    def link(self):
        """
        Linking must be done after attribute setup
        """
        #log.info("link")
        self.shader.link()  

    def use(self):
        #log.info("use")
        self.shader.use()  
    
    def unuse(self):
        #log.info("unuse")
        self.shader.unuse() 



 

if __name__ == '__main__':
    pass
