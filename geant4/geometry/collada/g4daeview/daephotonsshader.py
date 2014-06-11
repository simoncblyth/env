#!/usr/bin/env python
"""
When using position_name="position" DAEVertexBuffer does
the traditional glVertexPointer setup that furnishes gl_Vertex
to the shader. 


"""
from env.graphics.opengl.shader.shader import Shader
import logging
log = logging.getLogger(__name__)

vertex = r"""#version 110
// vertex : that simply passes through to geometry shader 

attribute vec4 %(position_name)s;
attribute vec4 momdir;
varying vec4 vMomdir;

void main()
{
    gl_Position = vec4( %(position_name)s.xyz, 1.) ; 
    //gl_Position = vec4( gl_Vertex.xyz , 1.) ; 

    vMomdir = 1000.*vec4( momdir.xyz, 1.) ;
    //vMomdir = vec4( 100.,100.,100., 1.) ;
}
"""

vertex_debug = r"""#version 110
// vertex_debug : for use without geometry shader 

attribute vec4 %(position_name)s;
attribute vec4 momdir;

void main()
{

    gl_Position = vec4( %(position_name)s.xyz, 1.) ; 
    //gl_Position = vec4( gl_Vertex.xyz , 1.) ; 
    gl_Position.xyz += 100.*momdir.xyz ; 
    gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
}
"""

geometry = r"""#version 120
#extension GL_EXT_geometry_shader4 : enable

//  http://www.opengl.org/wiki/Geometry_Shader_Examples

varying in vec4 vMomdir[] ; 

void main()
{
    // amplify primitive point into line

    gl_Position = gl_PositionIn[0];
    gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
    EmitVertex();

    gl_Position = gl_PositionIn[0];
    gl_Position.xyz += vMomdir[0].xyz ;

    gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
    EmitVertex();

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
    def __init__(self, dphotons ):

        ctx = { 'position_name':dphotons.position_name } 
        debug = dphotons.debug_shader
        if debug:
            log.debug("compiling debug shader")
            shader = Shader( vertex_debug , fragment, None, **ctx )
        else:
            log.debug("compiling normal shader")
            shader = Shader( vertex, fragment, geometry, **ctx )
        pass
        self.shader = shader

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
