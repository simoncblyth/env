# === func-gen- : graphics/thrust_opengl_interop/throgl fgp graphics/thrust_opengl_interop/throgl.bash fgn throgl fgh graphics/thrust_opengl_interop
throgl-src(){      echo graphics/thrust_opengl_interop/throgl.bash ; }
throgl-source(){   echo ${BASH_SOURCE:-$(env-home)/$(throgl-src)} ; }
throgl-vi(){       vi $(throgl-source) ; }
throgl-usage(){ cat << EOU

Interop between OpenGL/OptiX/Thrust/CUDA 
==========================================


harrism : NVIDIA expert on Interop (2011)
--------------------------------------------

* http://stackoverflow.com/questions/6481123/cuda-and-opengl-interop

As of CUDA 4.0, OpenGL interop is one-way. That means to do what you want (run
a CUDA kernel that writes data to a GL buffer or texture image), you have to
map the buffer to a device pointer, and pass that pointer to your kernel, as
shown in your example.

cudaGraphicsResourceGetMappedPointer is called every time display() is 
called because cudaGraphicsMapResource is called every frame.
Any time you re-map a resource you should re-get the mapped pointer, because it
may have changed. 

Why re-map every frame? 

Well, OpenGL sometimes moves buffer objects around in memory, 
for performance reasons (especially in memory-intensive GL applications). 

If you leave the resource mapped all the time, it can't do this, 
and performance may suffer. 

I believe GL's ability and need to virtualize memory objects is 
also one of the reasons the current GL interop API is one-way 
(the GL is not allowed to move CUDA allocations around,
and therefore you can't map a CUDA-allocated device pointer into a GL buffer
object).



Presentation on interop (uses deprecated API)
-----------------------------------------------

* http://www.nvidia.com/content/gtc/documents/1055_gtc09.pdf




EOU
}

throgl-env(){      elocal- ; }

throgl-sdir(){ echo $(env-home)/graphics/thrust_opengl_interop ; }
throgl-idir(){ echo $(local-base)/env/graphics/thrust_opengl_interop ; }
throgl-bdir(){ echo $(throgl-idir).build ; }
throgl-bindir(){ echo $(throgl-idir)/bin ; }

throgl-scd(){  cd $(throgl-sdir); }
throgl-cd(){   cd $(throgl-sdir); }

throgl-icd(){  cd $(throgl-idir); }
throgl-bcd(){  cd $(throgl-bdir); }
throgl-name(){ echo ThrustOpenGLInterop ; }

throgl-wipe(){
   local bdir=$(throgl-bdir)
   rm -rf $bdir
}

throgl-cmake(){
   local iwd=$PWD

   local bdir=$(throgl-bdir)
   mkdir -p $bdir
  
   throgl-bcd 
   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(throgl-idir) \
       $(throgl-sdir)

   cd $iwd
}


throgl-make(){
   local iwd=$PWD

   throgl-bcd
   make $*
   cd $iwd
}

throgl-install(){
   throgl-make install
}

throgl--()
{
    throgl-wipe
    throgl-cmake
    throgl-make
    throgl-install

}

