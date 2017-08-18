# === func-gen- : graphics/opengl/txf/txf fgp graphics/opengl/txf/txf.bash fgn txf fgh graphics/opengl/txf
txf-src(){      echo graphics/opengl/txf/txf.bash ; }
txf-source(){   echo ${BASH_SOURCE:-$(env-home)/$(txf-src)} ; }
txf-vi(){       vi $(txf-source) ; }
txf-usage(){ cat << EOU

OpenGL Transform Feedback
===========================


* http://prideout.net/blog/?tag=opengl-transform-feedback


Tutorial : sqrt via OpenGL 4.0+
---------------------------------

* https://open.gl/feedback


When you include a geometry shader, the transform feedback operation will
capture the outputs of the geometry shader instead of the vertex shader. For
example:

When using a geometry shader, the primitive specified to
glBeginTransformFeedback must match the output type of the geometry shader:


::

    simon:txf blyth$ TXF
    Frame::gl_init_window Renderer: NVIDIA GeForce GT 750M OpenGL Engine
    Frame::gl_init_window OpenGL version supported 4.1 NVIDIA-8.26.26 310.40.45f01
    1.000000
    2.000000
    3.000000
    1.414214
    2.414214
    3.414214
    1.732051
    2.732051
    3.732051
    2.000000
    3.000000
    4.000000
    2.236068
    3.236068
    4.236068





Ricci : 2010 OpenGL 4.0 review
-------------------------------

* http://www.g-truc.net/post-0269.html

Transform feedback is the OpenGL name given to Direct3D output stream. It
allows to capture processed vertices data before rasterisation and to be more
accurate, just before clipping. A first extension proposed by nVidia
(GL_NV_transform_feedback) has been promoted to GL_EXT_transform_feedback and
finally included in OpenGL 3.0 specification.

GL_ARB_transform_feedback2  : glDrawTransformFeedback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

...this extension provides the function glDrawTransformFeedback to use transform
feedback buffers as vertex shader source without having to query the primitives
written count. When querying this count with glGetQueryObjectuiv, the function
is going to stall the graphics pipeline waiting for the OpenGL commands to be
completed. glDrawTransformFeedback replaces glDrawArrays in this case and
doesn't need the vertices count, it's going to use automatically the count of
written primitives in the transform feedback object to draw.
GL_ARB_transform_feedback2 is part of OpenGL 4.0 but is also supported by
GeForce GT200 as an extension.



glBindBufferBase
-------------------

* https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glBindBufferBase.xhtml

::

    void glBindBufferBase(  GLenum target, GLuint index, GLuint buffer); 

    target
        Specify the target of the bind operation. target must be one of 

        * GL_ATOMIC_COUNTER_BUFFER (4.2+)
        * GL_TRANSFORM_FEEDBACK_BUFFER
        * GL_UNIFORM_BUFFER 
        * GL_SHADER_STORAGE_BUFFER. (4.3+)

    index
        Specify the index of the binding point within the array specified by target.

    buffer
        The name of a buffer object to bind to the specified binding point.





EOU
}

txf-env(){      elocal- ; } 

txf-sdir(){ echo $(env-home)/graphics/opengl/txf ; }
txf-idir(){ echo $(local-base)/env/graphics/opengl/txf ; }
txf-bdir(){ echo $(txf-idir).build ; }
txf-bindir(){ echo $(txf-idir)/bin ; }

txf-cd(){   cd $(txf-sdir); }
txf-c(){    cd $(txf-sdir); }
txf-icd(){  cd $(txf-idir); }
txf-bcd(){  cd $(txf-bdir); }

txf-name(){ echo TXF ; }

txf-wipe(){
   local bdir=$(txf-bdir)
   rm -rf $bdir
}

txf-cmake(){
   local iwd=$PWD

   local bdir=$(txf-bdir)
   mkdir -p $bdir
 
   opticks- 
 
   txf-bcd 
   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
       $(txf-sdir)

   cd $iwd
}

txf-make(){
   local iwd=$PWD

   txf-bcd
   make $*
   cd $iwd
}

txf-install(){
   txf-make install
}

txf--()
{
    txf-wipe
    txf-cmake
    txf-make
    txf-install
}


