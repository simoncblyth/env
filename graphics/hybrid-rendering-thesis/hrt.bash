# === func-gen- : graphics/hybrid-rendering-thesis/hrt fgp graphics/hybrid-rendering-thesis/hrt.bash fgn hrt fgh graphics/hybrid-rendering-thesis
hrt-src(){      echo graphics/hybrid-rendering-thesis/hrt.bash ; }
hrt-source(){   echo ${BASH_SOURCE:-$(env-home)/$(hrt-src)} ; }
hrt-vi(){       vi $(hrt-source) ; }
hrt-env(){      elocal- ; }
hrt-usage(){ cat << EOU

Example of use of OptiX with GLFW3
=====================================

Hybrid Rendering Thesis unsing OpenGL and Optix

Master's thesis at Narvik University College.

https://code.google.com/p/hybrid-rendering-thesis/


Physically based Rendering Background
--------------------------------------

* http://www.pbrt.org



Shared PBO
------------

Using shared PBO allows interop as OptiX can write directly 
into it, then can use standard OpenGL 

/usr/local/env/graphics/hrt/glfw_optix/src/triangle_scene.h::


    165         context->setEntryPointCount(1);
    166         context->setStackSize(2048);
    167         out_buffer_var = context->declareVariable("output_buffer");
    168         light_buffer = context->declareVariable("lights");
    ...
    196 
    197         /* Create shared GL/CUDA PBO */
    198         int element_size = 4 * sizeof(char);
    199         pbo = new Render::PBO(element_size * width * height, GL_STREAM_DRAW, true);
    200         out_buffer_obj = context->createBufferFromGLBO(RT_BUFFER_OUTPUT, pbo->getHandle() );
    201         out_buffer_obj->setFormat(RT_FORMAT_UNSIGNED_BYTE4);
    202         out_buffer_obj->setSize(width,height);
    203 
    204         out_buffer_var->set(out_buffer_obj);
    205 

    /// hmm just same as OptiXEngine::createOutputBuffer


env/pycuda/pycuda_pyopengl_interop/pixel_buffer.py
---------------------------------------------------

* http://www.songho.ca/opengl/gl_pbo.html


EOU
}
hrt-dir(){ echo $(local-base)/env/graphics/hrt ; }
hrt-cd(){  cd $(hrt-dir); }
hrt-mate(){ mate $(hrt-dir) ; }
hrt-get(){
   local dir=$(dirname $(hrt-dir)) &&  mkdir -p $dir && cd $dir

   svn checkout http://hybrid-rendering-thesis.googlecode.com/svn/trunk hrt

}
