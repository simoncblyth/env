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

SHADER = {}
SHADER['bit_sniffing'] = r"""

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

SHADER['vertex_for_geo'] = r"""// simply passes through to geometry shader 
#version 120
#extension GL_EXT_gpu_shader4 : require

uniform vec4  fparam; 
uniform ivec4 iparam; 

attribute vec4 position_time ;
attribute vec4 direction_wavelength;
attribute vec4 polarization_weight ;
attribute vec4 ccolor ; 

//attribute uvec4 flags;
//attribute ivec4 last_hit_triangle;

varying vec4 vMomdir;
varying vec4 vPoldir;
varying vec4 vColor ;


void main()
{
    gl_Position = vec4( position_time.xyz, 1.) ; 
    vMomdir = fparam.x*vec4( direction_wavelength.xyz, 1.) ;
    vPoldir = fparam.x*vec4( polarization_weight.xyz, 1.) ;
    vColor = ccolor ; 
}

"""

SHADER['vertex_no_geo'] = r"""//for use without geometry shader 
#version 120
#extension GL_EXT_gpu_shader4 : require

uniform vec4  fparam; 
uniform ivec4 iparam; 

attribute vec4 position_time ;
attribute vec4 direction_wavelength;
attribute vec4 ccolor ; 

varying vec4 fColor;


void main()
{
    gl_Position = vec4( position_time.xyz, 1.) ; 

    // scoot alpha zeros off to infinity and beyond
    if( ccolor.w == 0. ) gl_Position.w = 0. ;

    //gl_Position.xyz += fparam.x*direction_wavelength.xyz ; 
    gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
    fColor = ccolor ;


}

""" 

SHADER['geometry_point2line'] = r"""//amplify single vertex into two, generating line from point
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

SHADER['geometry_line2line'] = r"""
#version 120
#extension GL_EXT_geometry_shader4 : require
#extension GL_EXT_gpu_shader4 : require


varying in vec4 vMomdir[] ; 
varying in vec4 vPoldir[] ; 
varying in vec4 vColor[] ; 

varying out vec4 fColor ;

void main()
{
   gl_Position = gl_PositionIn[0];
   gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
   fColor = vColor[0] ;
   EmitVertex();

   gl_Position = gl_PositionIn[1];
   gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
   fColor = vColor[1] ;
   EmitVertex();

   EndPrimitive();
}

"""


SHADER['fragment_fcolor'] = r"""// minimal
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
    Properties

    #. shaderkey

    """
    def __init__(self, dphotons):
        self.dphotons = dphotons
        self._iparam = None
        self._fparam = None

        self._shadercfg = None
        self._shaderkey = None
        self.shaderkey = dphotons.cfg['shaderkey']   
        self.shader = self.make_shader()

    def shaderkey_changed(self, from_, to_):
        """
        Starting with one style, then changing to another causing invalid operations::

           g4daeview.sh --with-chroma --load 1 --style movie
           udp.py --style spagetti
        
        """
        log.info("shaderkey_changed %s => %s " % (from_,to_))
        self._shadercfg = None   # invalidate dependent, forcing recreation
        self.shader.cfg = self.shadercfg
        program = self.shader.program
        log.info("pull program %s into life " % program ) 

    def _get_shaderkey(self):
        return self._shaderkey 
    def _set_shaderkey(self, shaderkey):
        if shaderkey == self._shaderkey:return
        priorkey = self._shaderkey 
        self._shaderkey = shaderkey
        if not priorkey is None:
            self.shaderkey_changed(priorkey, self._shaderkey)
        pass
    shaderkey = property(_get_shaderkey, _set_shaderkey, doc="String controlling shader config ")

    def _get_shadercfg(self):
        if self._shadercfg is None:
            self._shadercfg = self.make_config(self.shaderkey) 
        return self._shadercfg 
    shadercfg = property(_get_shadercfg)

    def make_shader(self):
        shader = Shader( **self.shadercfg  )
        print shader
        return shader

    def make_config(self, shaderkey):
        """
        The input type to the geometry shader needs to be one of:

        * GL_POINTS
        * GL_LINES
        * GL_TRIANGLES
        * or ADJACENCY variants of LINES and TRIANGLES 

        NB that is the **input to the geometry shader**, which
        is distinct from the primitive type used in the draw call.

        For example when the geometry shader is expecting GL_LINES 
        primitives it is OK to pump GL_LINE_STRIP down the pipeline
        with the glDraw call.

        That was my expectation from the spec, but reality seems
        otherwise.

        #. line2line failing at Draw with invalid operation
           
        """
        cfg = {}
        cfg['shaderkey'] = shaderkey

        if shaderkey == "nogeo":

            cfg['vertex'] = "vertex_no_geo"
            cfg['fragment'] = "fragment_fcolor"
            cfg['geometry'] = None
            cfg['geometry_output_type'] = None

        elif shaderkey == "line2line":

            cfg['vertex'] = "vertex_for_geo"
            cfg['geometry'] = "geometry_line2line"
            cfg['geometry_input_type'] = "GL_LINES"   #  does not accept GL_LINE_STRIP, 
            cfg['geometry_output_type'] = "GL_LINE_STRIP"
            cfg['fragment'] = "fragment_fcolor"

        elif shaderkey == "point2line":

            cfg['vertex'] = "vertex_for_geo"
            cfg['geometry'] = "geometry_point2line"
            cfg['geometry_input_type'] = "GL_POINTS"
            cfg['geometry_output_type'] = "GL_LINE_STRIP"
            cfg['fragment'] = "fragment_fcolor"

        else:
            assert 0, "shader key %s not recognized " % shaderkey  
        pass
        
        for k,v in cfg.items():
            if not v is None and v[0:3] == 'GL_':
                cfg[k] = getattr(gl, cfg[k])  # promote strings starting GL_ to enum types

        log.info("%s cfg %s" % (self.__class__.__name__, repr(cfg)))
        for k in ['vertex','fragment','geometry']:
            if not cfg[k] is None:
                cfg[k] = "\n".join(["//%s" % cfg[k],SHADER[cfg[k]]])
            pass

        return cfg

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
