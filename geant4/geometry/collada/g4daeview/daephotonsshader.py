#!/usr/bin/env python

from env.graphics.opengl.shader.shader import Shader
import logging
log = logging.getLogger(__name__)

vertex = r"""
// simply pass through to geometry shader 

attribute %(momdir_type)s momdir;
varying %(momdir_type)s vMomdir;

void main()
{
    gl_Position = gl_Vertex ; 
    vMomdir = momdir ;
}
"""

vertex_debug = r"""
// for use without geometry shader 
attribute %(momdir_type)s momdir;
void main()
{
    gl_Position = gl_Vertex ; 
    gl_Position.xyz += momdir.xyz ; 
    gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
}
"""

geometry = r"""
#version 120
#extension GL_EXT_geometry_shader4 : enable

//  http://www.opengl.org/wiki/Geometry_Shader_Examples

varying in %(momdir_type)s vMomdir[] ; 

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
    gl_FragColor = vec4(1.0, 0, 0, 1);
}
"""

class DAEPhotonsShader(object):
    def __init__(self, **kwa):
        debug = kwa.pop('debug',False)
        if debug:
            log.info("compiling debug shader")
            shader = Shader( vertex_debug , fragment, None, **kwa )
        else:
            log.info("compiling normal shader")
            shader = Shader( vertex, fragment, geometry, **kwa )
        pass
        shader.link()
        self.shader = shader

    def bind(self):
        self.shader.bind()
    
    def unbind(self):
        self.shader.unbind()



 

if __name__ == '__main__':
    pass
