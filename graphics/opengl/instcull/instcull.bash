instcull-src(){      echo graphics/opengl/instcull/instcull.bash ; }
instcull-source(){   echo ${BASH_SOURCE:-$(env-home)/$(instcull-src)} ; }
instcull-vi(){       vi $(instcull-source) ; }
instcull-usage(){ cat << EOU
OpenGL Instance Culling
========================


Demo Project Exploring How Best to Arrange OpenGL Instance Culling Rendering
----------------------------------------------------------------------------------


Encapsulated Renderer Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

instElemUBO.cc
    exercise the encapsulated InstRenderer.hh allowing any Prim to be instanced
primRender.cc
    exercise the encapsulated Renderer.hh 


instcull1.cc
    monolith Demo class approach using modern attribute style of nature- with
    att props captured into the vertex array seems to be succeeding 
    to filter continuously 

    * http://rastergrid.com/blog/2010/02/instance-culling-using-geometry-shaders/

icdemo
    using non-monolithic ICDemo class, with CullShader, InstShader, SContext 



LOD ?
~~~~~~~~

* http://rastergrid.com/blog/2010/10/gpu-based-dynamic-geometry-lod/

 
 
Spell-it-out Renderer Tests for OpenGL usage debugging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

instanceMinimal.cc
    minimal instancing with glDrawArraysInstanced

instanceVA.cc
    just instancing with no culling derived from instance-
    demonstrates encapsulation provided by VertexArray, by combining 2 vbo A,B 
    and 2 sets of instance transforms I, J in four ways AI,AJ,BI,BJ
    Also shows UBO in operation.

oneCubeMinimal.cc
     non-encapsulated glDrawElements 

onetriangleMinimal.cc
    non-encapsulated glDrawArrays

onetriangleElementMinimal.cc
    non-encapsulated spell everything out rendering 
    using glDrawElements and UBO for matrix updating 
 

Unit Tests
~~~~~~~~~~~~

::

    BBTest.cc
    BoxTest.cc
    CamTest.cc
    CompTest.cc
    GeomTest.cc
    VueTest.cc







Refs
------

* :google:`opengl instance culling`

* http://www.geeks3d.com/20100210/opengl-3-2-geometry-instancing-culling-on-gpu-demo/
* https://www.gamedev.net/articles/programming/graphics/opengl-instancing-demystified-r3226/


Best source on transform feedback

* http://github.prideout.net/modern-opengl-prezo/

* https://github.com/OpenGLInsights/OpenGLInsightsCode
* https://github.com/Samsung/GearVRf/issues/89


aqnuep / rastergrid  (AMD driver developer) 
-------------------------------------------------

* https://www.opengl.org/discussion_boards/showthread.php/175530-Gemoetry-shader-view-frustum-culling

* http://rastergrid.com/blog/2010/02/instance-culling-using-geometry-shaders/
* http://rastergrid.com/blog/2010/06/instance-cloud-reduction-reloaded/
* http://rastergrid.com/blog/downloads/mountains-demo/


Buffer types other than GL_ARRAY_BUFFER for instance transforms ?

* http://rastergrid.com/blog/2010/01/uniform-buffers-vs-texture-buffers/


Dynamic Instanced LOD
---------------------------

* http://rastergrid.com/blog/2010/10/gpu-based-dynamic-geometry-lod/

Instead of the visibility filtering of instance transforms (like instcull- and nature-)
partition the visible instance transforms into three LOD streams according to distance.  
Thus updating GPU buffers of instance transforms for each LOD level.

Then at render just make three instanced draw calls, to show all visible instances
at their appropriate LOD level.



Approaches
------------


Hmm what about a visibility bitmask : small enough to go in uniforms or texture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


::

   18k bits ->  18000/8 = 2250 bytes
   36k bits ->  36000/8 = 4500 bytes


Its possible to stuff the requisite instance mask bits into a few MB of uniform or texture...

* BUT: Would the instance mask actually help though ?

* filtering via a mask would require a geometry shader

* seems like filtering instance transforms is best way (as of OpenGL 4.1) 
  as rendering pass is unchanged (hence pays no price)

* render just needs to grab instance transforms from the filtered buffer


How big for 18k, 36k transforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    sizeof(float) = 4 bytes
 
    4*4*4 = 64 bytes    per 4x4 matrix of floats

    64*18k = 1152 k  ~ 1.1M
    64*36k = 2304 k  ~ 2.3M



* https://stackoverflow.com/questions/29855353/how-to-know-max-buffer-storage-size-using-opengl-on-a-specific-device

::

    Frame::gl_init_window Renderer: NVIDIA GeForce GT 750M OpenGL Engine
    Frame::gl_init_window OpenGL version supported 4.1 NVIDIA-8.26.26 310.40.45f01
     name     GL_MAX_TEXTURE_BUFFER_SIZE val 134217728 val/1e6 134.218
     name      GL_MAX_UNIFORM_BLOCK_SIZE val 65536 val/1e6 0.065536


texel fetch faster than attrib access ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://stackoverflow.com/questions/37797240/why-is-texture-buffer-faster-than-vertex-inputs-when-using-instancing-in-glsl

::

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, size, buffer, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribDivisor(0, 1); // this makes the buffer instanced

    layout (location = 1) in vec3 perInstanceVector; // VBO instanced attribute
    outputVector = perInstanceVector;

::

    glBindTexture(GL_TEXTURE_BUFFER, textureVBO);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGB32F, VBO);

    uniform samplerBuffer textureBuffer; // texture buffer which has the same data as the previous VBO instanced attribute
    outputVector = texelFetch(textureBuffer, gl_InstanceID).xyz





UBO : GL_UNIFORM_BUFFER
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://www.geeks3d.com/20140704/gpu-buffers-introduction-to-opengl-3-1-uniform-buffers-objects/

From a GLSL shader point of view, an uniform buffer is a read-only memory buffer.


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
    #instcull-wipe
    instcull-cmake
    instcull-make
    instcull-install
}


