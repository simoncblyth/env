DAEPhotonsShader
=================

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



line strip finnicky
----------------------


::

     File "/usr/local/env/chroma_env/lib/python2.7/site-packages/env/geant4/geometry/collada/g4daeview/daephotons.py", line 179, in multidraw
        self.renderer.multidraw(slot=slot, counts=self.counts, firsts=self.firsts, drawcount=self.drawcount )
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/env/geant4/geometry/collada/g4daeview/daephotonsrenderer.py", line 162, in multidraw
        self.pbuffer.multidraw(mode=gl.GL_LINE_STRIP,  what='', drawcount=qcount, slot=slot, counts=counts, firsts=firsts) 
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/env/geant4/geometry/collada/g4daeview/daevertexbuffer.py", line 500, in multidraw
        gl.glMultiDrawArrays( gl.GL_LINE_STRIP, firsts, counts, drawcount )
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/OpenGL/platform/baseplatform.py", line 379, in __call__
        return self( *args, **named )
      File "errorchecker.pyx", line 50, in OpenGL_accelerate.errorchecker._ErrorChecker.glCheckError (src/errorchecker.c:854)
      OpenGL.error.GLError: GLError(
        err = 1282,
        description = 'invalid operation',
        baseOperation = glMultiDrawArrays,
        cArguments = (
            GL_LINE_STRIP,
            array([    0,    10,    20, ..., 4162...,
            array([3, 2, 2, ..., 9, 9, 9], dtype=...,
            4165,
        )
    )



shaders and integers dont mix in version 120
------------------------------------------------

::

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




