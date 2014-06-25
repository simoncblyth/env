DAEPhotonsShader
=================


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




