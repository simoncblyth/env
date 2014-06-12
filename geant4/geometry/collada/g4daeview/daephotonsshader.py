#!/usr/bin/env python
"""
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

import OpenGL.GL as gl


vertex = r"""#version 110
// vertex : that simply passes through to geometry shader 

uniform vec4 param; 

attribute vec4 position_weight;
attribute vec4 direction_wavelength;
attribute vec4 polarization_time;

varying vec4 vMomdir;

void main()
{
    gl_Position = vec4( position_weight.xyz, 1.) ; 
    vMomdir = param.x*vec4( direction_wavelength.xyz, 1.) ;
}
"""

vertex_debug = r"""#version 110
// vertex_debug : for use without geometry shader 

uniform vec4 param; 
attribute vec4 position_weight;
attribute vec4 direction_wavelength;

void main()
{
    gl_Position = vec4( position_weight.xyz, 1.) ; 
    gl_Position.xyz += param.x*direction_wavelength.xyz ; 
    gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
}
"""

geometry = r"""#version 120
#extension GL_EXT_geometry_shader4 : enable

//  http://www.opengl.org/wiki/Geometry_Shader_Examples

varying in vec4 vMomdir[] ; 
uniform int mode ; 

void main()
{

    gl_Position = gl_PositionIn[0];
    gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
    EmitVertex();

    // amplify primitive point into line
    if( mode > 0 ){
        gl_Position = gl_PositionIn[0];
        gl_Position.xyz += vMomdir[0].xyz ;
        gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
        EmitVertex();
    }

    if( mode > 1 ){
        gl_Position = gl_PositionIn[0];
        gl_Position.xyz += -2.*vMomdir[0].xyz ;
        gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
        EmitVertex();
    }


    EndPrimitive();
}
"""

fragment = r"""
void main()
{
    gl_FragColor = vec4(.0, 1.0, 0, 1);
}
"""

class DAEPhotonsShader(object):
    """
    NEXT:
    #. Try to reconfig the shader to do lines and points rather than having two shaders
    """
    def __init__(self, dphotons ):

        ctx = {}
        debug = dphotons.param.debugshader
        if debug:
            log.debug("compiling debug shader")
            shader = Shader( vertex_debug , fragment, None, **ctx )
        else:
            log.debug("compiling normal shader")
            #shader = Shader( vertex, fragment, geometry, geometry_output_type=gl.GL_POINTS )
            shader = Shader( vertex, fragment, geometry, geometry_output_type=gl.GL_LINE_STRIP )
        pass
        self.shader = shader
        self.dphotons = dphotons

    def set_param(self):
        """
        Hmm how to avoid calling this all the time before every draw ?

        TODO: try to make this need to be called only on a change 
        """
        param = self.dphotons.param.shader_uniform_param
        #log.info("set_param %s " % repr(param)) 
        self.shader.uniformf("param", *param)

    def set_mode(self, mode):
        self.shader.uniformi("mode", mode )

    def __str__(self):
        return "%s\n%s " % (self.__class__.__name__, str(self.shader))

    def link(self):
        """
        Linking must be done after attribute setup
        """
        self.shader.link()  ## LinkProgram 

    def bind(self):
        self.shader.bind()  ## UseProgram
    
    def unbind(self):
        self.shader.unbind() ## UseProgram(0)



 

if __name__ == '__main__':
    pass
