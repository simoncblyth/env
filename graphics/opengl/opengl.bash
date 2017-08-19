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


OSX Stuck At 4.1
------------------

* https://forum.unity3d.com/threads/opengl-4-3-or-apple-metal-gpu-offload-on-mac.396414/


18/08/2010 OpenGL 4.1 review
-------------------------------

* https://www.g-truc.net/post-0320.html

23/03/2010 OpenGL 4.0 review 
-------------------------------

* http://www.g-truc.net/post-0269.html


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
opengl-dir(){ echo $(local-base)/env/graphics/opengl/graphics/opengl-opengl ; }
opengl-cd(){  cd $(opengl-dir); }
opengl-mate(){ mate $(opengl-dir) ; }
opengl-get(){
   local dir=$(dirname $(opengl-dir)) &&  mkdir -p $dir && cd $dir

}
