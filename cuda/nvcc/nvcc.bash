# === func-gen- : cuda/nvcc/nvcc fgp cuda/nvcc/nvcc.bash fgn nvcc fgh cuda/nvcc
nvcc-src(){      echo cuda/nvcc/nvcc.bash ; }
nvcc-source(){   echo ${BASH_SOURCE:-$(env-home)/$(nvcc-src)} ; }
nvcc-vi(){       vi $(nvcc-source) ; }
nvcc-env(){      elocal- ; }
nvcc-usage(){ cat << EOU

NVCC Experience
=================

arch argument explanation
--------------------------

* http://stackoverflow.com/questions/17599189/what-is-the-purpose-of-using-multiple-arch-flags-in-nvidias-nvcc-compiler
* http://codeyarns.com/2014/03/03/how-to-specify-architecture-to-compile-cuda-code/


Opticks Flags
---------------

::

/usr/local/cuda/bin/nvcc -M -D__CUDACC__
       /Users/blyth/opticks/cudarap/CResource_.cu -o
             /usr/local/opticks/build/cudarap/CMakeFiles/CUDARap.dir//CUDARap_generated_CResource_.cu.o.NVCC-depend
    -ccbin
          /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang
         -m64 
         -DCUDARap_EXPORTS 
          -Xcompiler ,\"-fPIC\"
          -gencode=arch=compute_30,code=sm_30 
          -std=c++11 
          -O2 
          -DVERBOSE 
          --use_fast_math
          -m64 
          -DNVCC
           -I/usr/local/cuda/include 
           -I/Users/blyth/opticks/cudarap
           -I/usr/local/opticks/externals/plog/include
           -I/usr/local/cuda/include
           -I/Users/blyth/opticks/sysrap 
        nvcc fatal   : redefinition of argument 'machine'




OptiX 3.8.0 Programming Guide PDF, nvcc flag mentions 
-------------------------------------------------------

Sect 5.8 Extracts p60/61
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When nvcc is used, make sure the device code bitness is targeted by using 
the -m64 flag. The bitness of all PTX given to the OptiX API must be 64-bit.

When using nvcc to generate PTX output specify the -ptx flag. Note that any
host code in the CUDA file will not be present in the generated PTX file. Your
CUDA files should include <optix_world.h> to gain access to functions and
definitions required by OptiX and many useful operations for vector types and
ray tracing.

OptiX is not guaranteed to parse all debug information inserted by nvcc into
PTX files. We recommend avoiding the --device-debug nvcc flag. Note that this
flag is set by default on debug builds in Visual Studio.

In order to provide better support for compilation of PTX to different SM
targets, OptiX uses the .target information found in the PTX code to determine
compatibility with the currently utilized devices. If you wish your code to run
an sm_20 device, compiling the PTX with -arch sm_30 will generate an error even
if no sm_30 features are present in the code. Compiling to sm_20 will run on
sm_20 and higher targets.


Chapter 11.Performance Guidelines, p86
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When creating PTX code using nvcc, adding --use-fast-math as a compile option
can reduce code size and increase the performance for most OptiX programs. This
can come at the price of slightly decreased numerical floating point accuracy.
See the nvcc documentation for more details.


C++11
--------

* http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#axzz4Dokp2W5J

::

   --std c++11

Select a particular C++ dialect. 
The only value currently supported is c++11. 
Enabling C++11 mode also turns on C++11 mode for the host compiler.




CUDA runtime issue : "invalid device function" with sm_50 and sm_52  GPUs (Maxwell 1st and 2nd generation)
-----------------------------------------------------------------------------------------------------------

RESOLVED BY MOVING TO --arch sm_30

::

    GGeoViewTest --compute ## 

    2016-07-07 20:45:07.567 INFO  [23475] [OEngineImp::preparePropagator@170] OEngineImp::preparePropagator DONE 
    2016-07-07 20:45:07.567 INFO  [23475] [OpEngine::preparePropagator@102] OpEngine::preparePropagator DONE 
    2016-07-07 20:45:07.567 INFO  [23475] [OpSeeder::seedPhotonsFromGensteps@65] OpSeeder::seedPhotonsFromGensteps

    terminate called after throwing an instance of 'thrust::system::system_error'
      what():  function_attributes(): after cudaFuncGetAttributes: invalid device function

    Aborted (core dumped)
    [simonblyth@optix optixrap]$ 

    /// template snow storm /// 
    #24 0x00007ffff17c98bc in unsigned int TBuf::reduce<unsigned int>(unsigned int, unsigned int, unsigned int) const () from /home/simonblyth/local/opticks/lib/libThrustRap.so
    #25 0x00007ffff1260c20 in OpSeeder::seedPhotonsFromGenstepsImp (this=0x3fa35f0, s_gs=..., s_ox=...) at /home/simonblyth/opticks/opticksop/OpSeeder.cc:131
    #26 0x00007ffff1260b31 in OpSeeder::seedPhotonsFromGenstepsViaOptiX (this=0x3fa35f0) at /home/simonblyth/opticks/opticksop/OpSeeder.cc:114
    #27 0x00007ffff1260825 in OpSeeder::seedPhotonsFromGensteps (this=0x3fa35f0) at /home/simonblyth/opticks/opticksop/OpSeeder.cc:72
    #28 0x00007ffff12680c7 in OpEngine::seedPhotonsFromGensteps (this=0x18932e0) at /home/simonblyth/opticks/opticksop/OpEngine.cc:121
    #29 0x00007ffff07c277d in App::seedPhotonsFromGensteps (this=0x7fffffffd960) at /home/simonblyth/opticks/ggeoview/App.cc:987
    #30 0x0000000000403daf in main (argc=2, argv=0x7fffffffdb98) at /home/simonblyth/opticks/ggeoview/tests/GGeoViewTest.cc:110
        (gdb) 


* http://stackoverflow.com/questions/24067641/cuda-thrust-runtime-error-invalid-device-function

  ... Invalid device function often means that you have compiled for 
  a higher compute capability than the device you are trying to run on. ...
   ( Robert Crovella)


* https://developer.nvidia.com/cuda-gpus

::

                   Compute Capability
   Quadro M5000      5.2  
   Quadro M2000M     5.0


* NEED TO EXPERIMENT WITH NVCC FLAGS



maxwell compatibility guide 
------------------------------

* http://docs.nvidia.com/cuda/maxwell-compatibility-guide/#axzz4Djmddrut

1.4.2. Applications Using CUDA Toolkit 6.0, 6.5, or 7.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With version 6.0 of the CUDA Toolkit, nvcc can generate cubin files native to
the first-generation Maxwell architecture (compute capability 5.0); 

CUDA Toolkit 6.5 and later further add native support 
for second-generation Maxwell devices (compute capability 5.2). 

When using CUDA Toolkit 6.x or 7.0, to ensure that nvcc will generate cubin files 
for all recent GPU architectures as well as a PTX version for forward compatibility 
with future GPU architectures, specify the appropriate -gencode= parameters 
on the nvcc command line as shown in the examples below.


::

    /usr/local/cuda/bin/nvcc \
         -gencode=arch=compute_20,code=sm_20 \
         -gencode=arch=compute_30,code=sm_30 \
         -gencode=arch=compute_35,code=sm_35 \
         -gencode=arch=compute_50,code=sm_50 \
         -gencode=arch=compute_52,code=sm_52 \
         -gencode=arch=compute_52,code=compute_52 \
            -O2 -o mykernel.o -c mykernel.cu


Note that compute_XX refers to a PTX version and sm_XX refers to a cubin
version. The arch= clause of the -gencode= command-line option to nvcc
specifies the front-end compilation target and must always be a PTX version.
The code= clause specifies the back-end compilation target and can either be
cubin or PTX or both. Only the back-end target version(s) specified by the
code= clause will be retained in the resulting binary; at least one should be
PTX to provide compatibility with future architectures.


about arch and code
~~~~~~~~~~~~~~~~~~~~~~~

* https://codeyarns.com/2014/03/03/how-to-specify-architecture-to-compile-cuda-code/


*compute_XX*
         refers to the high level PTX, (that can be compiled to multiple specific GPUs) 
*sm_XX*
         refers to the GPU executable cubin (SASS) 


nvcc manual reference regards *--gencode*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#compilation-phases


This option provides a generalization of the --gpu-architecture=arch
--gpu-code=code,... option combination for specifying nvcc behavior with
respect to code generation. Where use of the previous options generates code
for different real architectures with the PTX for the same virtual
architecture, option --generate-code allows multiple PTX generations for
different virtual architectures. In fact, --gpu-architecture=arch
--gpu-code=code,... is equivalent to --generate-code arch=arch,code=code,....

--generate-code options may be repeated for different virtual architectures.


arch defaults 
~~~~~~~~~~~~~~~~~ 

* http://stackoverflow.com/questions/28932864/cuda-compute-capability-requirements

::

    CUDA VERSION   Min CC   Deprecated CC  Default CC
    5.5 (and prior) 1.0       N/A             1.0
    6.0             1.0       1.0             1.0
    6.5             1.1       1.x             2.0
    7.0             2.0       N/A             2.0
    7.5 (same as 7.0)

Min CC = minimum compute capability that can be specified to nvcc

Deprecated CC = If you specify this CC, you will get a deprecation message, but compile should still proceed.

Default CC = The architecture that will be targetted if no -arch or -gencode switches are used



Nice explanation from Robert Crovella
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://stackoverflow.com/questions/17599189/what-is-the-purpose-of-using-multiple-arch-flags-in-nvidias-nvcc-compiler

::

   CUDA C/C++ device code source --> PTX --> SASS

The virtual architecture (e.g. compute_20, whatever is specified by -arch
compute...) determines what type of PTX code will be generated. The additional
switches (e.g. -code sm_21) determine what type of SASS code will be generated.
SASS is actually executable object code for a GPU (machine language). An
executable can contain multiple versions of SASS and/or PTX, and there is a
runtime loader mechanism that will pick appropriate versions based on the GPU
actually being used.


thrust changelog, releases 
------------------------------

* https://github.com/thrust/thrust/releases
* https://github.com/thrust/thrust/blob/1.8.2/CHANGELOG

::

    Thrust v1.8.2 #628 CUDA's reduce_by_key fails on sm_50 devices   (1.8.2 ships with CUDA 7.5)
    Thrust v1.8.1 #628 CUDA's reduce_by_key fails on sm_50 devices   (1.8.1 ships with CUDA 7.0) 

thrust issue 628 
~~~~~~~~~~~~~~~~~~~

* https://github.com/thrust/thrust/issues/628
* see thrustrap-/tests/issue628Test.cu




thrust code organization
-----------------------------

Lots of Thrust will compile when in .cpp or .cu files BUT 
often what you get doesnt work how you want it to when compiled from .cpp 
(main example is optix/cuda/thrust interop).
Pragmatic solution to Thrust problems is to move 
as much as possible into .cu

Normally that means tedious development of headers that both compilers
can stomach to provide a bridge between the worlds.  But that is slow

Heterogenous C++ Class definition
-----------------------------------

By this I mean classes with some methods compiled by the host compiler
and some by the NVIDIA compiler.
The advantage is that do not need to laboriously bridge between the worlds
with separate headers (other than the  header of the class itself) 
can just use class members directly.

The bridging is using the implicit this parameter. 


boost issues
--------------

* http://stackoverflow.com/questions/8138673/why-does-nvcc-fails-to-compile-a-cuda-file-with-boostspirit

nvcc sometimes has trouble compiling complex template code such as is found in
Boost, even if the code is only used in __host__ functions.

When a file's extension is .cpp, nvcc performs no parsing itself and instead
forwards the code to the host compiler, which is why you observe different
behavior depending on the file extension.

If possible, try to quarantine code which depends on Boost into .cpp files
which needn't be parsed by nvcc.


hiding things from compilers
-----------------------------

Pimpl/opaque_pointer https://en.wikipedia.org/wiki/Opaque_pointer




EOU
}
nvcc-dir(){ echo $(local-base)/env/cuda/nvcc/cuda/nvcc-nvcc ; }
nvcc-cd(){  cd $(nvcc-dir); }
nvcc-mate(){ mate $(nvcc-dir) ; }
nvcc-get(){
   local dir=$(dirname $(nvcc-dir)) &&  mkdir -p $dir && cd $dir

}


nvcc-hello-(){ cat << EOH
#include <stdio.h>

int main() 
{
    printf( "running... $FUNCNAME\n" );
    return 0;
}
EOH
}


nvcc-hello(){
   local tmp=/tmp/$USER/env/cuda/nvcc/$FUNCNAME
   mkdir -p $tmp && cd $tmp

   local nam="hello"

   cuda- 


   $FUNCNAME- > $nam.cc
   nvcc $nam.cc -o $nam  && ./$nam

  
   which clang
   clang --version

}

nvcc-hello-notes(){ cat << EON

epsilon:nvcc-hello blyth$ nvcc-;nvcc-hello
nvcc fatal   : The version ('90100') of the host compiler ('Apple clang') is not supported

/usr/bin/clang
Apple LLVM version 9.1.0 (clang-902.0.39.1)
Target: x86_64-apple-darwin17.5.0
Thread model: posix
InstalledDir: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin
epsilon:nvcc-hello blyth$ 


epsilon:nvcc-hello blyth$ xcode-;xcode-92
sudo xcode-select --switch /Applications/Xcode/Xcode_9_2.app/Contents/Developer


epsilon:nvcc-hello blyth$ nvcc-;nvcc-hello
running... nvcc-hello-

/usr/bin/clang
Apple LLVM version 9.0.0 (clang-900.0.39.2)
Target: x86_64-apple-darwin17.5.0
Thread model: posix
InstalledDir: /Applications/Xcode/Xcode_9_2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin
epsilon:nvcc-hello blyth$ 




EON
}





