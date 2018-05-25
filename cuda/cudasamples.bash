# === func-gen- : cuda/cudasamples fgp cuda/cudasamples.bash fgn cudasamples fgh cuda
cudasamples-src(){      echo cuda/cudasamples.bash ; }
cudasamples-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cudasamples-src)} ; }
cudasamples-vi(){       vi $(cudasamples-source) ; }
cudasamples-env(){      elocal- ; }
cudasamples-usage(){ cat << EOU



* get impression that CUDA CMake lang support is not widely used 



CMake CUDA lang build gives error but Makefile build (and CMake cuda_add_executable does not)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    epsilon:build blyth$ ./simplePrintf 
    CUDA error at /usr/local/env/cuda_9_1_samples/common/inc/helper_cuda.h:1160 code=35(cudaErrorInsufficientDriver) "cudaGetDeviceCount(&device_count)" 
    epsilon:build blyth$ vi /usr/local/env/cuda_9_1_samples/common/inc/helper_cuda.h +1160

* https://devtalk.nvidia.com/default/topic/1027922/cuda-setup-and-installation/-solved-code-35-cudaerrorinsufficientdriver-error-on-mac-version-10-13-2-17c88-with-nvidia-geforce-gt-/

::

    https://devtalk.nvidia.com/default/topic/1031269/cuda-setup-and-installation/runtime-cudaerrorinsufficientdriver-from-cuda-9-1-sample-code/


Seems cannot target_link_libraries 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    -- Found CUDA: /Developer/NVIDIA/CUDA-9.1 (found version "9.1") 
    -- simplePrintf.CUDA_LIBRARIES : /Developer/NVIDIA/CUDA-9.1/lib/libcudart_static.a;-Wl,-rpath,/usr/local/cuda/lib 
    CMake Error at CMakeLists.txt:13 (target_link_libraries):
      The plain signature for target_link_libraries has already been used with
      the target "simplePrintf".  All uses of target_link_libraries with a target
      must be either all-keyword or all-plain.

      The uses of the plain signature are here:

       * /opt/local/share/cmake-3.11/Modules/FindCUDA.cmake:1867 (target_link_libraries)










EOU
}
cudasamples-dir(){ echo $(local-base)/env/cuda_9_1_samples ; }
cudasamples-tbase(){ echo /tmp/env/cuda_9_1_samples ; }
cudasamples-cd(){  cd $(cudasamples-dir); }

cudasamples-sample(){ echo 0_Simple/simplePrintf ;  }

cudasamples-sdir(){ echo $(cudasamples-dir)/$(cudasamples-sample) ; }
cudasamples-bdir(){ echo $(cudasamples-tbase)/$(cudasamples-sample).build ; }

cudasamples-scd(){  cd $(cudasamples-sdir) ; }
cudasamples-bcd(){  cd $(cudasamples-bdir) ; }

cudasamples-info(){ cat << EOI

   cudasamples-sample : $(cudasamples-sample)
   cudasamples-sdir   : $(cudasamples-sdir)
   cudasamples-bdir   : $(cudasamples-bdir)

EOI
}

cudasamples-cmake(){ 
    local iwd=$PWD

    local sdir=$(cudasamples-sdir)
    local bdir=$(cudasamples-bdir)

    rm -rf $bdir
    mkdir -p $bdir && cd $bdir 

    cmake -DCUDASAMPLES_DIR=$(cudasamples-dir)  $sdir

    cd $iwd
}


cudasamples--()
{
    cudasamples-cmake
    cudasamples-bcd

    make 
}



