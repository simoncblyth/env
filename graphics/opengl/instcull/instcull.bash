instcull-src(){      echo graphics/opengl/instcull/instcull.bash ; }
instcull-source(){   echo ${BASH_SOURCE:-$(env-home)/$(instcull-src)} ; }
instcull-vi(){       vi $(instcull-source) ; }
instcull-usage(){ cat << EOU
OpenGL Instance Culling
========================


Best source on transform feedback

* http://github.prideout.net/modern-opengl-prezo/



Projection Matrix Tricks
---------------------------

* http://www.terathon.com/gdc07_lengyel.pdf




* https://gamedev.stackexchange.com/questions/83161/draw-selected-instances-of-vao-gldrawarraysinstanced

Keep in mind that various instancing features have been added to OpenGL at different times:

OpenGL 3.1 ARB_draw_instanced and ARB_uniform_buffer_object become core, allowing basic instanced rendering using gl_InstanceID.
OpenGL 3.3 ARB_instanced_arrays becomes core, allowing for instanced vertex attributes through glVertexAttribDivisor.
OpenGL 4.2 ARB_base_instance makes instanced attributes more flexible. ARB_transform_feedback_instanced allows for instanced drawing from transform feedback buffers.
OpenGL 4.3 ARB_vertex_attrib_binding becomes core, adding glVertexBindingDivisor, which works like glVertexAttribDivisor but operates on any attributes that use a particular buffer binding index rather than just one attribute index. Also, ARB_shader_storage_buffer_object, while not directly related to instancing, solves some issues with ARB_uniform_buffer_object.
OpenGL 4.5 ARB_direct_state_access becomes core, including glVertexArrayBindingDivisor, the DSA version of glVertexBindingDivisor.



glDrawTransformFeedbackInstanced (not in OSX OpenGL 4.1 core or extensions)
----------------------------------------------------------------------------------------

Anyhow it does not seem to be whats needed for instanced culling...

HUH THATS WIERD : glDrawTransformFeedbackInstanced  ASSUMES THE TRANSFORM FEEDBACK 
IS WRITING THE GEOMETRY RATHER THAN THE INSTANCE TRANSFORMS... 
CLEARLY IT IS BETTER TO CULL BY SELECTING INSTANCE TRANSFORMS 
AS 4x4 MATRICES OR 4x1 OFFSETS ARE GOING TO BE MUCH SMALLER 
THAN THE ACTUAL VERTICES


* https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_transform_feedback_instanced.txt

* https://www.khronos.org/opengl/wiki/GLAPI/glDrawTransformFeedbackInstanced

* https://www.g-truc.net/doc/OpenGL%204%20Hardware%20Matrix%202014-05.pdf

  * table of OSX OpenGL 4.1 extensions is very small


::

    void glDrawTransformFeedbackInstanced(GLenum mode, GLuint id, GLsizei instancecount);

    mode
        what kind of primitives to render
    id
        name of a transform feedback object from which to retrieve a primitive count

    instancecount
        number of instances of the geometry to render.


glDrawTransformFeedbackInstanced draws multiple copies of a range of primitives
of a type specified by mode using a count retrieved from the transform
feedback stream specified by stream of the transform feedback object
specified by id. 

Calling glDrawTransformFeedbackInstanced is equivalent to calling glDrawArraysInstanced with 

* mode and primcount as specified, 
* first set to zero, 
* and count set to the number of vertices captured on vertex stream zero 
  the last time transform feedback was active on the transform feedback object named by id.

Calling glDrawTransformFeedbackInstanced is equivalent to calling
glDrawTransformFeedbackStreamInstanced with stream set to zero.


::

    void glDrawArraysInstanced(  GLenum mode, GLint first, GLsizei count, GLsizei primcount); 

    mode 
        kind of primitives to render
    first  
        starting index in the enabled arrays
    count
        number of indices to be rendered
    primcount
        number of instances of the specified range of indices to be rendered 



EOU
}

instcull-env(){      elocal- ; } 

instcull-sdir(){ echo $(env-home)/graphics/opengl/instcull ; }
instcull-idir(){ echo $(local-base)/env/graphics/opengl/instcull ; }
instcull-bdir(){ echo $(instcull-idir).build ; }
instcull-bindir(){ echo $(instcull-idir)/bin ; }

instcull-cd(){   cd $(instcull-sdir); }
instcull-c(){    cd $(instcull-sdir); }
instcull-icd(){  cd $(instcull-idir); }
instcull-bcd(){  cd $(instcull-bdir); }

instcull-name(){ echo INSTCULL ; }

instcull-wipe(){
   local bdir=$(instcull-bdir)
   rm -rf $bdir
}

instcull-cmake(){
   local iwd=$PWD

   local bdir=$(instcull-bdir)
   mkdir -p $bdir
 
   opticks- 
 
   instcull-bcd 
   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
       $(instcull-sdir)

   cd $iwd
}

instcull-make(){
   local iwd=$PWD

   instcull-bcd
   make $*
   cd $iwd
}

instcull-install(){
   instcull-make install
}

instcull--()
{
    instcull-wipe
    instcull-cmake
    instcull-make
    instcull-install
}


