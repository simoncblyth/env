#!/usr/bin/env python
"""
"""

import logging
log = logging.getLogger(__name__)

import OpenGL.GL as gl
from env.graphics.opengl.shader.shader import Shader
from env.graphics.color.wav2RGB import wav2RGB_glsl

SHADER = {}
SHADER['vertex_no_geo'] = r"""//for use without geometry shader, eg by spagetti and confetti styles
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
    gl_PointSize = ( ccolor.w < 0. ) ? 9. : 4. ; 

    if( ccolor.w == 0. ) gl_Position.w = 0. ;    // scoot alpha zeros off to infinity and beyond

    gl_Position = gl_ModelViewProjectionMatrix * gl_Position;

    fColor = vec4( ccolor.xyz, abs(ccolor.w) )  ; 

}
""" 




SHADER['vertex_for_geo'] = r"""// simply passes through to geometry shader 
#version 120
#extension GL_EXT_geometry_shader4 : require
#extension GL_EXT_gpu_shader4 : require

uniform vec4  fparam; 
uniform ivec4 iparam; 

attribute vec4 position_time ;
attribute vec4 direction_wavelength;
attribute vec4 polarization_weight ;
attribute vec4 ccolor ; 

varying vec4 vMomdir;
varying vec4 vPoldir;
varying vec4 vColor ;

void main()
{
    // pass attributes without transformation 
    gl_Position = vec4( position_time.xyz, 1.) ; 
    gl_PointSize = ( ccolor.w < 0. ) ? 9. : 4. ; 

    vMomdir = vec4( fparam.x*direction_wavelength.xyz, 1.) ;
    vPoldir = vec4( fparam.x*polarization_weight.xyz, 1.) ;
    vColor = vec4( ccolor.xyz, abs(ccolor.w) )  ; 

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

        gl_PointSize = gl_PointSizeIn[0];
        gl_Position = gl_PositionIn[0];

        gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
        fColor = vColor[0] ;
        EmitVertex();


        gl_PointSize = gl_PointSizeIn[0];
        gl_Position = gl_PositionIn[0];
        gl_Position.xyz += vMomdir[0].xyz ;

        gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
        fColor = vColor[0] ;

        EmitVertex();

     /*
        gl_PointSize = gl_PointSizeIn[0];
        gl_Position = gl_PositionIn[0];
        gl_Position.xyz += vMomdir[0].xyz ;
        gl_Position.xyz += vPoldir[0].xyz ;
        gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
        fColor = vColor[0] ;
        EmitVertex();
      */

        EndPrimitive();

    }
}
"""

SHADER['geometry_point2point'] = r"""
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

       gl_PointSize = gl_PointSizeIn[0];
       fColor = vColor[0] ;

       EmitVertex();
       EndPrimitive();
   }
}

"""


SHADER['fragment_fcolor'] = r"""// minimal
#version 120
#extension GL_EXT_geometry_shader4 : require
#extension GL_EXT_gpu_shader4 : require

varying vec4 fColor ;

void main()
{
    gl_FragColor = fColor ;
}
"""



class DAEPhotonsShader(object):
    """
    Initially tried swapping in and out shaders with deletions etc.. 
    this approach had issues with some transitions not working. Changing 
    the approach to keeping the shaders in a registry and swapping between
    them as needed has proved to work, with no switching problems.
    """
    shaderkeys = ['nogeo','p2p','p2l',]
    def __init__(self, dphotons):

        shadercfg = {} 
        for key in self.shaderkeys:
            shadercfg[key] = self.make_config(key)
        pass
        self.shadercfg = shadercfg 
        self.shaders = {}

        self.dphotons = dphotons
        self._iparam = None
        self._fparam = None

        self._shaderkey = None
        self.shaderkey = None # dphotons.cfg['shaderkey']   

    def get_shader(self, key ):
        """
        Provides shader for the key, making it if necessary 

        :param key: shader key 
        :return: Shader instance
        """
        assert key in self.shaderkeys
        if not key in self.shaders:
            self.shaders[key] = Shader( **self.shadercfg[key] )
        return self.shaders[key]

    shader = property(lambda self:self.get_shader(self.shaderkey))

    def _get_shaderkey(self):
        return self._shaderkey 
    def _set_shaderkey(self, shaderkey):
        if shaderkey == self._shaderkey:return
        self._shaderkey = shaderkey
    shaderkey = property(_get_shaderkey, _set_shaderkey, doc="String controlling shader config ")


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
            cfg['geometry'] = None
            cfg['fragment'] = "fragment_fcolor"

            cfg['geometry_input_type'] = None
            cfg['geometry_output_type'] = None

        elif shaderkey == "p2p":

            cfg['vertex'] = "vertex_for_geo"
            cfg['geometry'] = "geometry_point2point"
            cfg['fragment'] = "fragment_fcolor"

            cfg['geometry_input_type'] = "GL_POINTS"
            cfg['geometry_output_type'] = "GL_POINTS"

        elif shaderkey == "p2l":

            cfg['vertex'] = "vertex_for_geo"
            cfg['geometry'] = "geometry_point2line"
            cfg['fragment'] = "fragment_fcolor"

            cfg['geometry_input_type'] = "GL_POINTS"
            cfg['geometry_output_type'] = "GL_LINE_STRIP"

        else:
            assert 0, "shader key %s not recognized " % shaderkey  
        pass
        
        for k,v in cfg.items():
            if not v is None and v[0:3] == 'GL_':
                cfg[k] = getattr(gl, cfg[k])  # promote strings starting GL_ to enum types

        log.debug("%s cfg %s" % (self.__class__.__name__, repr(cfg)))
        for k in ['vertex','fragment','geometry']:
            if not cfg[k] is None:
                cfg[k] = "\n".join(["//%s" % cfg[k],SHADER[cfg[k]]])
            pass

        return cfg

    def update_uniforms(self):
        #log.info("update_uniforms")
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
        if fparam == self._fparam:
            #log.info("_set_fparam unchanged %s " % repr(fparam))
            return 
        #log.info("_set_fparam %s " % repr(fparam))
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
