DAEPhotonsShader
=================

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

