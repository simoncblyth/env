#!/usr/bin/env python

from env.graphics.opengl.shader.shader import Shader
import logging
log = logging.getLogger(__name__)

vertex = r"""
attribute vec3 momdir;
varying vec3 vMomdir;

void main()
{
    gl_Position = gl_Vertex ; 
    vMomdir = momdir ;

    //gl_Position.xyz += momdir ; 
    //gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
}
"""

geometry = r"""
#version 120
#extension GL_EXT_geometry_shader4 : enable

//  http://www.opengl.org/wiki/Geometry_Shader_Examples

varying in vec3 vMomdir[] ; 

void main()
{
    // from a point into a line

    gl_Position = gl_PositionIn[0];
    gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
    EmitVertex();

    gl_Position = gl_PositionIn[0];
    gl_Position.xyz += vMomdir[0] ;

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
    def __init__(self):
        shader = Shader( vertex, fragment, geometry )
        shader.link()
        self.shader = shader

    def bind(self):
        self.shader.bind()
    
    def unbind(self):
        self.shader.unbind()
 

if __name__ == '__main__':
    pass
