#!/usr/bin/env python

from env.graphics.opengl.shader.shader import Shader
import logging
log = logging.getLogger(__name__)


vertex = r"""
void main()
{
    gl_Position = gl_ModelViewProjectionMatrix*gl_Vertex ; 
}
"""

geometry = None
_geometry = r"""
#version 120
#extension GL_EXT_geometry_shader4 : enable
void main()
{
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
