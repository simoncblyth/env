# === func-gen- : graphics/opengl/opengl fgp graphics/opengl/opengl.bash fgn opengl fgh graphics/opengl
opengl-src(){      echo graphics/opengl/opengl.bash ; }
opengl-source(){   echo ${BASH_SOURCE:-$(env-home)/$(opengl-src)} ; }
opengl-vi(){       vi $(opengl-source) ; }
opengl-env(){      elocal- ; }
opengl-usage(){ cat << EOU

OPENGL
=======


* http://www.openglsuperbible.com/2013/12/09/vertex-array-performance/


* http://stackoverflow.com/questions/18814977/using-a-vbo-to-draw-lines-from-a-vector-of-points-in-opengl
* https://www.opengl.org/discussion_boards/showthread.php/176296-glDrawElements-multiple-calls-one-index-array
* http://www.opengl.org/wiki/Vertex_Buffer_Object#Vertex_Buffer_Object

Tute
------

* https://alfonse.bitbucket.io/oldtut/Basics/Intro%20What%20is%20OpenGL.html

* http://github.prideout.net/modern-opengl-prezo/


Samples Pack / Driver Bugs 
---------------------------

* https://www.opengl.org/sdk/docs/tutorials/OGLSamples/
* http://www.g-truc.net/project-0026.html
* http://www.g-truc.net/post-0373.html

OpenGL Test Suites, Piglit 
---------------------------

* https://archive.fosdem.org/2015/schedule/event/gl_testing/attachments/slides/670/export/events/attachments/gl_testing/slides/670/slides.pdf


GLSL Common Mistakes
-----------------------

* https://www.khronos.org/opengl/wiki/GLSL_:_common_mistakes


OpenGL Ref Pages
------------------

* https://www.khronos.org/registry/OpenGL-Refpages/

OpenGL Super Bible
---------------------

* http://www.openglsuperbible.com
* http://apprize.info/programming/opengl_1/index.html

glQueryIndexed Suspected Bug
--------------------------------

Interesting to see an OpenGL implementation

* https://www.mesa3d.org/intro.html

* https://patchwork.freedesktop.org/patch/33317/
* https://cgit.freedesktop.org/mesa/mesa/tree/src/mesa/drivers/dri/i965
* https://cgit.freedesktop.org/mesa/mesa/tree/src/mesa/drivers/dri/i965/gen6_queryobj.c


Am I misunderstanding glQueryIndexed ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://apprize.info/programming/opengl_1/13.html

openGL can only count and add up results into one query object at a time, but
it can manage several query objects and perform many queries back-to-back.


OpenGL is modeled as a pipeline and can have many things going on at the same
time. If you draw something simple such as a bounding box, it’s likely that
won’t have reached the end of the pipeline by the time you need the result of
your query. This means that when you call glGetQueryObjectuiv(), your
application may have to wait a while for OpenGL to finish working on your
bounding box before it can give you the answer and you can act on it.

In our next example, we render ten bounding boxes before we ask for the result
of the first query. This means that OpenGL’s pipeline can be filled, and it can
have a lot of work to do and is therefore much more likely to have finished
working on the first bounding box before we ask for the result of the first
query. In short, the more time we give OpenGL to finish working on what we’ve
asked it for, the more likely it is that it’ll have the result of your query
and the less likely it is that your application will have to wait for results.
Some complex applications take this to the extreme and use the results of
queries from the previous frame to make decisions about the new frame.



Querying without stalling ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


http://apprize.info/programming/opengl_1/13.html
     
querying will likely stall the pipeline
to avoid that could check if the result is available 
first with GL_QUERY_RESULT_AVAILABLE
before making the actual query ...
     
The outcome of applyFork is updated instance buffers for 
each LOD and corresponding counts..   Need to know
the counts to properly use these buffers.
     
       
But how to organize deferred querying ?
    
Hmm would be complicated... would need to have a 2nd set of
instance buffers and ping-pong between them ?
    
Actually could simply not launch the culling for every frame...
it does not matter if LOD piles/culling are a bit behind
    


OSX OpenGL Driver Bugs
-------------------------

* http://renderingpipeline.com/2012/07/macos-x-opengl-driver-bugs/ 


Books
-------

ogli
    https://github.com/OpenGLInsights/OpenGLInsightsCode


OSX Stuck At 4.1
------------------

* https://apple.stackexchange.com/questions/213892/why-doesnt-os-x-require-graphics-card-driver-updates

* https://forum.unity3d.com/threads/opengl-4-3-or-apple-metal-gpu-offload-on-mac.396414/

* https://support.apple.com/en-us/HT202823


18/08/2010 OpenGL 4.1 review
-------------------------------

* https://www.g-truc.net/post-0320.html

23/03/2010 OpenGL 4.0 review 
-------------------------------

* http://www.g-truc.net/post-0269.html


What State is held in VAO ?
------------------------------

* https://gamedev.stackexchange.com/questions/99236/what-state-is-stored-in-an-opengl-vertex-array-object-vao-and-how-do-i-use-the

glVertexAttribPointer
-------------------------

* https://stackoverflow.com/questions/17149728/when-should-glvertexattribpointer-be-called

Stores the offset of the attribute data with the buffer object to be used for that
attribute (as well as format information and stride data for it). – Nicol Bolas (Jun 17 '13 at 15:26)


GL_UNIFORM_BUFFER
-----------------------

* http://www.geeks3d.com/20140704/gpu-buffers-introduction-to-opengl-3-1-uniform-buffers-objects/

The main advantage of using uniform buffers is that they can be shared between
several GLSL shaders. Then, a single UBO is enough for all shaders that use the
same data.

From a GLSL shader point of view, an uniform buffer is a read-only memory buffer.


Nicol Bolas
~~~~~~~~~~~~~~~~

* https://stackoverflow.com/questions/10020679/how-does-glgetuniformblockindex-know-whether-to-look-in-the-vertex-shader-or-the

Assuming the program has been fully linked, it doesn't matter. Here are the
possibilities for glGetUniformBlockIndex and their consequences:

* The uniform block name given is not in any of the shaders. Then you get back
GL_INVALID_INDEX.

* The uniform block name given is used in one of the shaders. Then you get back
the block index for it, to be used with glUniformBlockBinding.

* The uniform block name given is used in multiple shaders. Then you get back
the block index that means all of them, to be used with glUniformBlockBinding.

The last part is key. If you specify a uniform block in two shaders, with the
same name, then GLSL requires that the definitions of those uniform blocks be
the same. If they aren't, then you get an error in glLinkProgram. Since they
are the same (otherwise you wouldn't have gotten here), GLSL considers them to
be the same uniform block.

Even if they're in two shader stages, they're still the same uniform block. So
you can share uniform blocks between stages, all with one block binding. As
long as it truly is the same uniform block.



opengl workflow
-------------------

* https://stackoverflow.com/questions/17149728/when-should-glvertexattribpointer-be-called

The function glVertexAttribPointer specifies the format and source buffer
(ignoring the deprecated usage of client arrays) of a vertex attribute that is
used when rendering something (i.e. the next glDraw... call).

Now there are two scenarios. You either use vertex array objects (VAOs) or you
don't (though not using VAOs is deprecated and discouraged/prohibited in modern
OpenGL). If you're not using VAOs, then you would usually call
glVertexAttribPointer (and the corresponding glEnableVertexAttribArray) right
before rendering to setup the state properly. If using VAOs though, you
actually call it (and the enable function) inside the VAO creation code (which
is usually part of some initialization or object creation), since its settings
are stored inside the VAO and all you need to do when rendering is bind the VAO
and call a draw function.

But no matter when you call glVertexAttribPointer, you should bind the
corresponding buffer right before (no matter when that was actually created and
filled), since the glVertexAttribPointer function sets the currently bound
GL_ARRAY_BUFFER as source buffer for this attribute (and stores this setting,
so afterwards you can freely bind another VBO).

So in modern OpenGL using VAOs (which is recommended), it's usually similar to
this workflow:


::

    //initialization
    glGenVertexArrays
    glBindVertexArray

    glGenBuffers
    glBindBuffer
    glBufferData

    glVertexAttribPointer
    glEnableVertexAttribArray

    glBindVertexArray(0)

    glDeleteBuffers //you can already delete it after the VAO is unbound, since the
                    //VAO still references it, keeping it alive (see comments below).

    ...

    //rendering
    glBindVertexArray
    glDrawWhatever





GLSL Compatibility
---------------------

* https://stackoverflow.com/questions/26266198/glsl-invalid-call-of-undeclared-identifier-texture2d

Cripes. Finally found the answer right after I posted the question. texture2D has been replaced by texture.

Yes, be aware that on OS X #version 150 can only mean #version 150 core. On
other platforms where compatibility profiles are implemented, you can continue
to use things that were deprecated beginning in GLSL 1.30 such as texture2D if
you write #version 150 compatibility. You really don't want that, but it's
worth mentioning ;) – Andon M. Coleman






Excellent Sources
------------------

* https://open.gl/


Primitive Restart
-------------------

GL_PRIMITIVE_RESTART

Enables primitive restarting. If enabled, any one of the draw commands which
transfers a set of generic attribute array elements to the GL will restart the
primitive when the index of the vertex is equal to the primitive restart index.
See glPrimitiveRestartIndex.


GL_CULL_FACE

If enabled, cull polygons based on their winding in window coordinates. See glCullFace.

* https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glCullFace.xhtml

Specifies whether front- or back-facing facets are candidates for culling.
Symbolic constants GL_FRONT, GL_BACK, and GL_FRONT_AND_BACK are accepted. The
initial value is GL_BACK.



Transform Feedback
-------------------

* txf-


Blogs
-------

* http://prideout.net/blog/?cat=3

Lightweight GUI ?
-------------------

* :google:`OpenGL Lightweight GUI`

  * https://code.google.com/p/glui2/wiki/Screenshots  its C++ 


16 bit float ? intended for storage, not computation
------------------------------------------------------

* http://en.wikipedia.org/wiki/Half-precision_floating-point_format
* 1-5-10 sign-exponent-fraction bit layout

objective : pack photon data optimally 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Photon counts ~3M, steps to restain ~10 

* want to be able to efficiently hold ~30M photon records
* one float4 quad is 32*4 = 8*4*4 = 128 bits = 16 bytes 
* one half4 quad is 16*4 = 4*4*4 = 64 bits = 8 bytes

Positions can be scaled to fit in -1.f:1.f box, and time into 0.f:1.f

* use OpenGL normalized signed short for these, so prep a signed short int 
  to represent the float, from CUDA make_short4(  

* CUDA short4 


SHRT_MAX

#define SHRT_MAX 32767 
#define SHRT_MIN (-32768)




* need to convert floats from OptiX CUDA to populate the structure

* 16 bytes*30M (480 MB) should be just doable... with 2GB GPU memory 
* need to record more than just position+time want flag bits too
* drop the direction, as next step gives that anyhow
* drop polarization and weight 
* cut precision of floats from 32->16 bits

::

    rbuffer[record_offset+0] = make_short4( p.position.x,    p.position.y,    p.position.z,     p.time );


* https://www.opengl.org/registry/specs/NV/gpu_shader5.txt


refs
~~~~~

* http://www.igorsevo.com/Article.aspx?article=Million+particles+in+CUDA+and+OpenGL


opengl normalized integer
~~~~~~~~~~~~~~~~~~~~~~~~~

* https://www.opengl.org/wiki/Normalized_Integer

A Normalized Integer is an integer which is used to store a decimal floating
point number. When formats use such an integer, OpenGL will automatically
convert them to/from floating point values as needed. This allows normalized
integers to be treated equivalently with floating-point values, acting as a
form of compression.


populate record from OptiX/CUDA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* __device__ unsigned short __float2half_rn(float x)  

* half seems inconvenient, creating signed integers to be OpenGL normalized into -1.f:1.f
  allows to define the record quad as 









workflow
~~~~~~~~~~

* allocate array for VBO from C++, eg npy-/NumpyEvt::setGenstepData
* describe content via MultiViewNPY (formerly MultiVecNPY)

  * arrange in quads eg float4 4*32bit=128bit vpos vdir vpol iflg 
    note some flexibility the 4th quad is treated as ivec4 in shaders despite being 
    declared as float4 

  * at C++/C/CUDA/OptiX level can union trick to write int/uint into float
    so types do not matter : just keep quad arrangement for GPU efficient 
    handling and simple 3D array addressing from NumPy and NPY

* upload and setup vertex attributes based on MultiViewNPY with oglrap-/Rdr::upload
  (NB this is tied to a particular shader pipeline eg "gl/pos" for photon presentation)

* attributes accessed from glsl shaders via:: 

   layout(location = 0) in vec3  vpos;
   layout(location = 2) in vec3  vpol;
   layout(location = 3) in ivec4 iflg; 

* is glsl constrained to whole quad having same type ? perhaps not floatBitsToInt/floatBitsToUint

  * https://www.opengl.org/sdk/docs/man4/html/floatBitsToInt.xhtml
  * https://www.opengl.org/sdk/docs/man4/html/intBitsToFloat.xhtml
  * https://www.opengl.org/wiki/Layout_Qualifier_(GLSL)
  * can two different attributes describe same location ?

    * yes, using *component* qualifier,  but cannot mix types it seems


* buffer setup in OptiX context eg optixrap-/OptiXEngine::initGenerate::

    354     NPY* photons = evt->getPhotonData();
    356     int photon_buffer_id = photons ? photons->getBufferId() : -1 ;
    ...
    359         unsigned int photon_count = photons->getShape(0);
    360         unsigned int photon_numquad = photons->getShape(1);
    361         unsigned int photon_totquad = photon_count * photon_numquad ;
    ...
    366         m_photon_buffer = m_context->createBufferFromGLBO(RT_BUFFER_INPUT_OUTPUT, photon_buffer_id);
    367         m_photon_buffer->setFormat(RT_FORMAT_FLOAT4);
    368         m_photon_buffer->setSize( photon_totquad );
    369         m_context["photon_buffer"]->set( m_photon_buffer );

* buffer contents access from OptiX programs via array addressing using launch thread index

cu/generate.cu::

    rtBuffer<float4>    photon_buffer;

cu/photon.h::

     57 __device__ void psave( Photon& p, optix::buffer<float4>& pbuffer, unsigned int photon_offset )
     58 {
     59     pbuffer[photon_offset+0] = make_float4( p.position.x,    p.position.y,    p.position.z,     p.time );








GL_HALF_FLOAT GLhalf
~~~~~~~~~~~~~~~~~~~~~~

* https://www.opengl.org/wiki/Small_Float_Formats


Sub-16 bit floats ?
~~~~~~~~~~~~~~~~~~~~~

* https://www.opengl.org/wiki/Vertex_Specification_Best_Practices

Normals
    The precision of normals usually isn't that important. And since normalized
    vectors are always on the range [-1, 1], its best to use a Normalized Integer
    format of some kind. The three components of a normal can be stored in a single
    32-bit integer via the GL_INT_2_10_10_10_REV type. You can ignore the last,
    2-bit component, or you can find something useful to stick into it.

Colors
    can be stored in normalized GL_UNSIGNED_BYTEs, so a single color can be packed
    into 4 bytes. If you need more color precision, GL_UNSIGNED_INT_2_10_10_10_REV
    is available, with 2 bits for alpha. If you absolutely need HDR colors, you can
    make use of GL_R11F_G11F_B10F, assuming the Float Precision works out. If not,
    you can employ GL_HALF_FLOATs instead of the expense of GL_FLOAT.

Positions

    These are fairly hard to pack more efficiently than GL_FLOAT, but this depends
    on your data and how much work you're willing to do. You can employ
    GL_HALF_FLOAT, but remember the range and precision limits relative to 32-bit
    floats.  
 
    A time-tested alternative is to use normalized GLshorts. To do this, 
    you rearrange your model space data so that all positions are packed in a
    [-1, 1] box around the origin. You do that by finding the min/max values in XYZ
    among all positions. Then you subtract the center point of the min/max box from
    all vertex positions; followed by scaling all of the positions by half the
    width/height/depth of the min/max box. You need to keep the center point and
    scaling factors around.  When you build your model-to-view matrix (or
    model-to-whatever matrix), you need to apply the center point offset and scale
    at the top of the transform stack (so at the end, right before you draw). Note
    that this offset and scale should not be applied to normals, as they have a
    separate model space.


normalize ?
~~~~~~~~~~~~~~

For glVertexAttribPointer, if normalized is set to GL_TRUE, it indicates that
values stored in an integer format are to be mapped to the range [-1,1] (for
signed values) or [0,1] (for unsigned values) when they are accessed and
converted to floating point. Otherwise, values will be converted to floats
directly without normalization.

::

    glVertexAttribPointer(index, size, type, normalize, stride, pointer);


* http://stackoverflow.com/questions/11678986/using-glshort-instead-of-glfloat-for-vertices


NVIDIA half 
~~~~~~~~~~~~

Are tied to NVIDIA for OptiX, so using NV specific
features in shaders not much of an additional restriction.

* https://www.opengl.org/wiki/GLSL_:_nVidia_specific_features
* http://http.download.nvidia.com/developer/presentations/GDC_2004/gdc_2004_NV_GLSL.pdf

  * very old presentation, still recommended ?


::

     #ifndef __GLSL_CG_DATA_TYPES
     # define half2 vec2
     # define half3 vec3
     # define half4 vec4
     #endif


GLM half
~~~~~~~~~~~

* https://www.opengl.org/discussion_boards/archive/index.php/t-185557.html
* hvec types removed in GLM 0.9.5 (2013/12/25)
* http://www.g-truc.net/post-0618.html

CUDA half
~~~~~~~~~~

* http://developer.download.nvidia.com/compute/cuda/4_1/rel/toolkit/docs/online/group__CUDA__MATH__INTRINSIC__CAST_g40476d89168be1daf8f03411027037ad.html

::

    __device__ unsigned short __float2half_rn(float x)  

    Convert the single-precision float value x to a half-precision 
    floating point value represented in unsigned short format, 
    in round-to-nearest-even mode.

CPU half
~~~~~~~~~

* http://half.sourceforge.net



EOU
}
opengl-dir(){ echo $(local-base)/env/graphics/opengl/ogl-samples ; }
opengl-fold(){ echo $(dirname $(opengl-dir)) ; }
opengl-cd(){  cd $(opengl-dir); }
opengl-c(){  cd $(opengl-dir); }
opengl-get(){
   local dir=$(dirname $(opengl-dir)) &&  mkdir -p $dir && cd $dir

   #local url=https://downloads.sourceforge.net/project/ogl-samples/OpenGL%20Samples%20Pack%204.1.7.2/ogl-samples-4.1.7.2.zip
   local url=https://github.com/g-truc/ogl-samples/releases/download/4.5.3.0/ogl-samples-4.5.3.0.zip
   [ ! -f $(basename $url) ] && curl -L -O $url 
   [ ! -d ogl-samples ] && unzip $(basename $url)  

}

opengl-ref-get()
{ 
   cd $(opengl-fold) 
   local url=https://www.khronos.org/registry/OpenGL/specs/gl/glspec41.core.pdf
   [ ! -f $(basename $url) ] && curl -L -O $url
}

opengl-ref(){ open $(opengl-fold)/${1:-glspec41.core.pdf} ; }


