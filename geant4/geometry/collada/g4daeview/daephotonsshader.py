#!/usr/bin/env python
"""

OpenGL Shader Language
=======================

GLSL 120
----------

* :google:`glsl 120 spec`

  * http://www.opengl.org/registry/doc/GLSLangSpec.Full.1.20.8.pdf

GLSL 120 Extensions
---------------------

* http://stackoverflow.com/questions/15107521/opengl-extensions-how-to-use-them-correctly-in-c-and-glsl

GL_EXT_gpu_shader4
~~~~~~~~~~~~~~~~~~~~~~

* https://www.opengl.org/registry/specs/EXT/gpu_shader4.txt

  * `uvec4`
  * bitwise operators
  * int/uint attributes

GL_EXT_geometry_shader4
~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://www.opengl.org/registry/specs/EXT/geometry_shader4.txt
* http://www.opengl.org/wiki/Geometry_Shader_Examples

  * vertex and primitive generation in geometry stage between vertex and fragment

glVertexAttribIPointer API hunt
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Trying to get integers attributes into shader proving difficult::

    delta:OpenGL blyth$ find . -name '*.py' -exec grep -H glVertexAttribIPointer {} \;
    ./platform/entrypoint31.py:glVertexAttribIPointer
    ./raw/GL/NV/vertex_program4.py:def glVertexAttribIPointerEXT( index,size,type,stride,pointer ):pass
    ./raw/GL/VERSION/GL_3_0.py:def glVertexAttribIPointer( index,size,type,stride,pointer ):pass
    delta:OpenGL blyth$ 


* http://developer.download.nvidia.com/opengl/specs/GL_NV_vertex_program4.txt

::

    In [15]: import OpenGL.raw.GL.VERSION.GL_3_0  as g30

    In [16]: g30.glVertexAttribIPointer?
    Type:       glVertexAttribIPointer
    String Form:<OpenGL.platform.baseplatform.glVertexAttribIPointer object at 0x10babc290>
    File:       /usr/local/env/chroma_env/lib/python2.7/site-packages/OpenGL/platform/baseplatform.py
    Definition: g30.glVertexAttribIPointer(self, *args, **named)
    Docstring:  <no docstring>
    Call def:   g30.glVertexAttribIPointer(self, *args, **named)




GL_NV_vertex_program4
~~~~~~~~~~~~~~~~~~~~~~~~

* trying to use this extension from glsl gives not supported
* http://developer.download.nvidia.com/opengl/specs/GL_NV_vertex_program4.txt

::

    In [11]: import OpenGL.raw.GL.NV.vertex_program4 as nv4

    In [12]: nv4.glVertexAttribIPointerEXT
    Out[12]: <OpenGL.platform.baseplatform.glVertexAttribIPointerEXT at 0x10bb9fed0>

    In [13]: nv4.EXTENSION_NAME  
    Out[13]: 'GL_NV_vertex_program4'




Fixed Pipeline and Shaders
---------------------------

When using position_name="position" DAEVertexBuffer does
the traditional glVertexPointer setup that furnishes gl_Vertex
to the shader. 

Legacy way prior to move to generic attributes::

    gl_Position = vec4( gl_Vertex.xyz , 1.) ; 
    //vMomdir = vec4( 100.,100.,100., 1.) ;



Bitwise operations glsl
-------------------------

* http://www.geeks3d.com/20100831/shader-library-noise-and-pseudo-random-number-generator-in-glsl/


* http://stackoverflow.com/questions/2182002/convert-big-endian-to-little-endian-in-c-without-using-provided-func


"""

import logging
log = logging.getLogger(__name__)

import OpenGL.GL as gl
from env.graphics.opengl.shader.shader import Shader
from env.graphics.color.wav2RGB import wav2RGB_glsl


bit_sniffing = r"""

// making integers useful inside glsl 120 + GL_EXT_gpu_shader4 is too much effort yo be worthwhile
//  problems 
//    #. cannot find symbol glVertexAttribIPointer
//    #. uint type not working so cannot do proper  

    uvec4 cf ;
    int nb = 0 ;
    int mb = -1 ;
    for( int n=0 ; n < 32 ; ++n ){
          cf.x = ( 1 << n ) ;
          if (( TEST & cf.x ) != 0){
                nb += 1 ;
                mb = n ;
          }
    }

    if      (mb==29) vColor = vec4( 1.0, 0.0, 0.0, 1.0);
    else if (mb==30) vColor = vec4( 0.0, 1.0, 0.0, 1.0);
    else if (mb==31) vColor = vec4( 0.0, 0.0, 1.0, 1.0);
    else             vColor = vec4( 1.0, 1.0, 1.0, 1.0);

    uvec4 b = uvec4( 0xff, 0xff00, 0xff0000,0xff000000) ;
    uvec4 r = uvec4( TEST >> 24 , TEST >> 8, TEST << 8 , TEST << 24 );
    uvec4 t ;
    t.x = ( r.x & b.x ) | ( r.z & b.z ) | ( r.y & b.y ) | ( r.w & b.w );
"""

vertex = r"""// vertex : that simply passes through to geometry shader 
#version 120
#extension GL_EXT_gpu_shader4 : require

uniform vec4  fparam; 
uniform ivec4 iparam; 

attribute vec4 position_weight;
attribute vec4 direction_wavelength;
attribute vec4 polarization_time;
attribute vec4 ccolor ; 
attribute uvec4 flags;
attribute ivec4 last_hit_triangle;

varying vec4 vMomdir;
varying vec4 vPoldir;
varying vec4 vColor ;


void main()
{
    gl_Position = vec4( position_weight.xyz, 1.) ; 
    vMomdir = fparam.x*vec4( direction_wavelength.xyz, 1.) ;
    vPoldir = fparam.x*vec4( polarization_time.xyz, 1.) ;
    vColor = ccolor ; 
}

"""

vertex_debug = r"""// vertex_debug : use without geometry shader 
#version 120
#extension GL_EXT_gpu_shader4 : require

uniform vec4  fparam; 
uniform ivec4 iparam; 

attribute vec4 position_weight;
attribute vec4 direction_wavelength;
attribute vec4 ccolor ; 

varying vec4 fColor;


void main()
{
    gl_Position = vec4( position_weight.xyz, 1.) ; 
    //gl_Position.xyz += fparam.x*direction_wavelength.xyz ; 
    gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
    fColor = ccolor ;
}

""" % { 'funcs':wav2RGB_glsl }

geometry = r"""//  geometry : amplify single vertex into two, generating line from point
#version 120
#extension GL_EXT_geometry_shader4 : require
#extension GL_EXT_gpu_shader4 : require

varying in vec4 vMomdir[] ; 
varying in vec4 vPoldir[] ; 
varying in vec4 vColor[] ; 

varying out vec4 fColor ;

uniform  vec4 fparam; 
uniform ivec4 iparam; 

void main()
{
    // dont emit the primitive for alpha 0.
    if( vColor[0].w > 0. ){

        gl_Position = gl_PositionIn[0];
        gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
        fColor = vColor[0] ;
        EmitVertex();

        gl_Position = gl_PositionIn[0];
        gl_Position.xyz += vMomdir[0].xyz ;
        gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
        fColor = vColor[0] ;
        EmitVertex();

        gl_Position = gl_PositionIn[0];
        gl_Position.xyz += vMomdir[0].xyz ;
        gl_Position.xyz += vPoldir[0].xyz ;
        gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
        fColor = vColor[0] ;
        EmitVertex();

        EndPrimitive();

    }
}
"""

fragment = r"""// fragment : minimal
#version 120
#extension GL_EXT_gpu_shader4 : require

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

    def update_uniforms(self):
        self.iparam = self.dphotons.param.shader_iparam
        self.fparam = self.dphotons.param.shader_fparam

    def _get_iparam(self):
        return self._iparam
    def _set_iparam(self, iparam):
        if iparam == self._iparam:return 
        self._iparam = iparam
        self.shader.uniformi("iparam", *self._iparam)
    iparam = property(_get_iparam, _set_iparam, doc="shader iparam uniform ")

    def _get_fparam(self):
        return self._fparam
    def _set_fparam(self, fparam):
        if fparam == self._fparam:return 
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
