# === func-gen- : geometry/vecgeom/vecgeom fgp geometry/vecgeom/vecgeom.bash fgn vecgeom fgh geometry/vecgeom
vecgeom-src(){      echo geometry/vecgeom/vecgeom.bash ; }
vecgeom-source(){   echo ${BASH_SOURCE:-$(env-home)/$(vecgeom-src)} ; }
vecgeom-vi(){       vi $(vecgeom-source) ; }
vecgeom-usage(){ cat << EOU

VecGeom : The vectorized geometry library for particle-detector simulation (toolkits).
========================================================================================

Sandro Christian Wenzel

* https://gitlab.cern.ch/VecGeom/VecGeom


TODO: look into VecGeom boolean handling
-------------------------------------------

* https://gitlab.cern.ch/VecGeom/VecGeom/blob/master/volumes/SpecializedBooleanVolume.h



VecGeom CUDA
-------------

::

    simon:VecGeom blyth$ find . -type f | wc -l
        1109
    simon:VecGeom blyth$ find . -type f -exec grep -l CUDA {} \; | wc -l
         241



Presentations
---------------


Sandro Wenzel : Towards a high performance geometry library for particle-detector simulation (ACAT 2014)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://indico.cern.ch/event/258092/contributions/1588561/attachments/454205/629615/ACAT2014GeometryTalkNewStyleV2.pdf
* ~/opticks_refs/ACAT2014GeometryTalkNewStyleV2.pdf


GeantV Geometry: SIMD abstraction and interfacing with CUDA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Johannes de Fine Licht (johannes.definelicht@cern.ch)
Sandro Wenzel (sandro.wenzel@cern.ch)

* http://indico.cern.ch/event/289682/contributions/664274/attachments/540916/745663/johannes_concurrency_forum_3.pdf
* ~/opticks_refs/johannes_concurrency_forum_3.pdf


inline namespaces
-------------------

* http://stackoverflow.com/questions/11016220/what-are-inline-namespaces-for

* https://msdn.microsoft.com/en-us/library/5cb46ksf.aspx

Inline namespaces (C++ 11)

In contrast to an ordinary nested namespace, members of an inline namespace are
treated as members of the parent namespace. This characteristic enables
argument dependent lookup on overloaded functions to work on functions that
have overloads in a parent and a nested inline namespace. I

You can use inline namespaces as a versioning mechanism to manage changes to
the public interface of a library. For example, you can create a single parent
namespace, and encapsulate each version of the interface in its own namespace
nested inside the parent. The namespace that holds the most recent or preferred
version is qualified as inline, and is therefore exposed as if it were a direct
member of the parent namespace. Client code that invokes the Parent::Class will
automatically bind to the new code. Clients that prefer to use the older
version can still access it by using the fully qualified path to the nested
namespace that has that code.  The inline keyword must be applied to the first
declaration of the namespace in a compilation unit.


source/benchmarking/Benchmarker.c::

     01 /// \file Benchmarker.cu
     02 /// \author Johannes de Fine Licht
     03 
     04 #include "benchmarking/Benchmarker.h"
     05 
     06 #include "base/Stopwatch.h"
     07 #include "backend/cuda/Backend.h"
     08 #include "management/CudaManager.h"
     09 
     10 namespace vecgeom {
     11 inline namespace cuda {
     12 
     13 __global__ void ContainsBenchmarkCudaKernel(VPlacedVolume const *const volume, const SOA3D<Precision> positions,
     14                                             const int n, bool *const contains)
     15 {
     16   const int i = ThreadIndex();
     17   if (i >= n) return;
     18   contains[i] = volume->Contains(positions[i]);
     19 }
     20 
     21 __global__ void InsideBenchmarkCudaKernel(VPlacedVolume const *const volume, const SOA3D<Precision> positions,
     22                                           const int n, Inside_t *const inside)
     23 {
     24   const int i = ThreadIndex();
     25   if (i >= n) return;
     26   inside[i] = volume->Inside(positions[i]);
     27 }




namespace cuda
----------------

::

    simon:VecGeom blyth$ find . -type f -exec grep -H namespace\ cuda {} \;
    ./backend/cuda/Backend.h:    namespace cuda {
    ./backend/cuda/Interface.h:inline namespace cuda {
    ./backend/cuda/Interface.h:namespace cuda {
    ./base/Cuda.h:  #define VECGEOM_DEVICE_FORWARD_DECLARE(X)  namespace cuda { X }  class __QuietSemi
    ./base/Cuda.h:     namespace cuda { classOrStruct X; }                               \
    ./base/Cuda.h:     namespace cuda { template <ArgType Arg> classOrStruct X; }         \
    ./base/Cuda.h:     namespace cuda { template <ArgType1 Arg1,ArgType2 Arg2> classOrStruct X; }        \
    ./base/Cuda.h:     namespace cuda { namespace NS { classOrStruct X; } }                      \
    ./base/Cuda.h:     namespace cuda { template <ArgType1 Arg1,ArgType2 Arg2> classOrStruct X; }                \
    ./base/Cuda.h:     namespace cuda { template <ArgType1 Arg1,ArgType2 Arg2,ArgType3 Arg3> classOrStruct X; }              \
    ./base/Cuda.h:     namespace cuda { template <ArgType1 Arg1,ArgType2 Arg2,ArgType3 Arg3> classOrStruct X; }              \
    ./base/Cuda.h:     namespace cuda { template <ArgType1 Arg1,ArgType2 Arg2,ArgType3 Arg3> classOrStruct X; }                \
    ./base/Cuda.h:    namespace cuda { template <ArgType1 Arg1,ArgType2 Arg2,ArgType3 Arg3,ArgType4 Arg4> classOrStruct X; }                 \
    ./base/Cuda.h:     namespace cuda { namespace NS { classOrStruct Def; } }                      \
    ./base/Cuda.h:     namespace cuda { template <ArgType1 Arg1,ArgType2 Arg2> classOrStruct X; }                \
    ./base/Cuda.h:     namespace cuda { template <ArgType1 Arg1,ArgType2 Arg2,ArgType3 Arg3> classOrStruct X; }              \
    ./base/Cuda.h:     namespace cuda { template <ArgType1 Arg1,ArgType2 Arg2,ArgType3 Arg3> classOrStruct X; }              \
    ./base/Cuda.h:     namespace cuda { template <ArgType1 Arg1,ArgType2 Arg2,ArgType3 Arg3> classOrStruct X; }                \
    ./base/Cuda.h:     namespace cuda { template <ArgType1 Arg1,ArgType2 Arg2,ArgType3 Arg3,ArgType4 Arg4> classOrStruct X; }                \
    ./base/Map.h:namespace cuda {
    ./base/Scale3D.h:namespace cuda {
    ./base/Transformation3D.h:namespace cuda {
    ./source/backend/cuda/Interface.cpp:} // End namespace cuda
    ./source/benchmarking/Benchmarker.cu:inline namespace cuda {
    ./source/benchmarking/NavigationBenchmarker.cu:inline namespace cuda {
    ./source/benchmarking/NavigationBenchmarker.cu:} // end of namespace cuda
    ./source/CudaManager.cpp:namespace cuda {
    ./source/CudaManager.cu:inline namespace cuda {
    ./source/CudaManager_0.cu:inline namespace cuda {
    ./source/Vector.cpp:inline namespace cuda {
    ./VecCore/include/VecCore/CUDA.h:#define VECCORE_DECLARE_CUDA(T) T; namespace cuda { T; }
    simon:VecGeom blyth$ 


VecCore/include/VecCore/CUDA.h
     VECCORE_DECLARE_* macros branched definitions




Can VecGeom be made to work within OptiX programs ?
-----------------------------------------------------

If it can, then:

* could avoid manual step of writing analytic geometry code
* BUT it is not a panacea, often the transition from volume to surface based geometry
  needed some cleanups of things like touching volumes that cause coincident surfaces
  which have to be fixed
* highly non-trivial, 


::

    simon:VecGeom blyth$ vecgeom-cmake
    -- No build type selected, default to Release
    -- Configuring with "Scalar" backend.
    -- Compiling with CUDA enabled
    -- Found CUDA: /usr/local/cuda (found version "7.0") 
    -- Could NOT find Vc: found neither VcConfig.cmake nor vc-config.cmake (Required is at least version "1.2.0")
    -- Vc version 1.2.0 was not found
    -- Found PythonInterp: /opt/local/bin/python (found version "2.7.11") 
    -- Looking for pthread.h
    -- Looking for pthread.h - found
    -- Looking for pthread_create
    -- Looking for pthread_create - found
    -- Found Threads: TRUE  
    -- Compiling for NATIVE SIMD architecture
    -- Compiling with C++ flags:  -O3 -DNDEBUG  -stdlib=libc++ -Wall -fPIC -ffast-math -ftree-vectorize -march=native  -DVECGEOM_SCALAR -DVECGEOM_CUDA -DVECGEOM_CUDA_NO_VOLUME_SPECIALIZATION -DVECGEOM_QUADRILATERALS_VC -DVECGEOM_NO_SPECIALIZATION -DVECGEOM_INPLACE_TRANSFORMATIONS -DVECGEOM_USE_INDEXEDNAVSTATES  -march=native
    -- Compiling with NVCC flags: -std=c++11;-Xcompiler;-Wno-unused-function;-Xcudafe;--diag_suppress=code_is_unreachable;-Xcudafe;--diag_suppress=initialization_not_reachable
    -- EXCLUDING ABBoxManager FROM CUDA
    -- EXCLUDING HybridManager2 FROM CUDA
    -- EXCLUDING ABBoxNavigator FROM CUDA
    -- EXCLUDING Medium FROM CUDA
    -- EXCLUDING NavigationSpecializer FROM CUDA
    -- EXCLUDING ResultComparator FROM CUDA
    -- Testing with CTest enabled.
    -- Downloading data files
    -- found existing file cms2015.root
    --  doing nothing ... 
    -- found existing file ExN03.root
    --  doing nothing ... 
    -- Configuring done
    -- Generating done
    -- Build files have been written to: /usr/local/env/geometry/VecGeom.build



Cannot access issues
---------------------

Issues are hidden behind CERN SSO which fails::

    Microsoft.IdentityServer.Web.RequestFailedException: MSIS7000: The sign in request is not compliant to the WS-Federation language for web browser clients or the SAML 2.0 protocol WebSSO profile.
    Reference number: e3d0ecc5-2ea2-4fea-befa-dba4121ddb3f


This issue may be related to the one I get sometimes with IHEP SSO.  Follow that up in sso-


Intrinsic header compilation error : WORKAROUND comment out the offending header 
-----------------------------------------------------------------------------------

VecCore/Common include of SIMD.h/x86intrin.h causes lots of compiler errors
with nvcc from CUDA 7.0 

::

    simon:VecGeom.build blyth$ vecgeom--
    [  1%] Building NVCC (Device) object CMakeFiles/vecgeomcuda.dir/source/vecgeomcuda_generated_CudaManager.cu.o
    /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/../lib/clang/6.0/include/__wmmintrin_aes.h(35): error: identifier "__builtin_ia32_aesenc128" is undefined
   ...

    Error limit reached.
    100 errors detected in the compilation of "/var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T//tmpxft_00002c57_00000000-7_CudaManager.cpp1.ii".

    make[2]: *** [CMakeFiles/vecgeomcuda.dir/source/vecgeomcuda_generated_CudaManager.cu.o] Error 1
    make[1]: *** [CMakeFiles/vecgeomcuda_static.dir/all] Error 2
    make: *** [all] Error 2


nvcc error messages do not provide the header include heirarchy, so had to 
do some detective work to find the header trail::

    delta:VecGeom blyth$ git status
    ...
    Untracked files:
      (use "git add <file>..." to include in what will be committed)

        VecCore/include/VecCore/Common_0.h
        VecCore/include/VecCore/SIMD_0.h
        VecCore/include/VecCore/VecCore_0
        base/Global_0.h
        management/CudaManager_0.h
        source/CudaManager_0.cu
        test/cmstestdata/cms2015.root


nvcc had problems compiling some clang intel intrinsic headers

* http://clang.llvm.org/doxygen/immintrin_8h_source.html


Commenting inclusion of x86intrin.h gets further
---------------------------------------------------

From build.log get lots of nvlink messages::

    0411 [ 51%] Building CXX object CMakeFiles/vecgeom.dir/source/CudaManager.cpp.o
     412 [ 52%] Building CXX object CMakeFiles/vecgeom.dir/source/backend/cuda/Interface.cpp.o
     413 [ 52%] Building CXX object CMakeFiles/vecgeom.dir/source/GeoManager.cpp.o
     414 [ 52%] Building CXX object CMakeFiles/vecgeom.dir/source/CppExporter.cpp.o
     415 [ 52%] Building CXX object CMakeFiles/vecgeom.dir/source/benchmarking/NavigationBenchmarker.cpp.o
     416 [ 53%] Linking CXX static library libvecgeom.a
     417 [ 53%] Built target vecgeom
     418 [ 53%] Building NVCC intermediate link file CMakeFiles/vecgeomcuda.dir/vecgeomcuda_intermediate_link.o
     419 nvlink warning : Stack size for entry function '_ZN7vecgeom4cuda30CudaManagerPrintGeometryKernelEPKNS0_13VPlacedVolumeE' cannot be statically determined
     420 nvlink info    : Function '_ZNK7vecgeom4cuda31ScalarShapeImplementationHelperINS0_22PolyconeImplementationILin1ELin1EEEE8ContainsERKNS0_8Vector3DIdEE' has address taken but no possible call to it
     421 nvlink info    : Function '_ZNK7vecgeom4cuda31ScalarShapeImplementationHelperINS0_22PolyconeImplementationILin1ELin1EEEE8ContainsERKNS0_8Vector3DIdEERS6_' has address taken but no possible call to it
     422 nvlink info    : Function '_ZNK7vecgeom4cuda31ScalarShapeImplementationHelperINS0_22PolyconeImplementationILin1ELin1EEEE16UnplacedContainsERKNS0_8Vector3DIdEE' has address taken but no possible call to it
     423 nvlink info    : Function '_ZNK7vecgeom4cuda31ScalarShapeImplementationHelperINS0_22PolyconeImplementationILin1ELin1EEEE6InsideERKNS0_8Vector3DIdEE' has address taken but no possible call to it
     ...
     855 nvlink info    : Function '_ZNK7vecgeom4cuda31ScalarShapeImplementationHelperINS0_21BooleanImplementationIL16BooleanOperation2ELin1ELin1EEEE19PlacedDistanceToOutERKNS0_8Vector3DIdEES9_d' has address taken but no possible call to it
     856 nvlink info    : Function '_ZNK7vecgeom4cuda31ScalarShapeImplementationHelperINS0_21BooleanImplementationIL16BooleanOperation2ELin1ELin1EEEE10SafetyToInERKNS0_8Vector3DIdEE' has address taken but no possible call to it
     857 nvlink info    : Function '_ZNK7vecgeom4cuda31ScalarShapeImplementationHelperINS0_21BooleanImplementationIL16BooleanOperation2ELin1ELin1EEEE11SafetyToOutERKNS0_8Vector3DIdEE' has address taken but no possible call to it
     858 Scanning dependencies of target vecgeomcuda
     859 [ 53%] Linking CXX shared library libvecgeomcuda.so
     860 [ 86%] Built target vecgeomcuda
     861 [ 86%] Building NVCC intermediate link file CMakeFiles/cudauserlib.dir/cudauserlib_intermediate_link.o
     862 nvlink warning : Stack size for entry function '_ZN7vecgeom4cuda30CudaManagerPrintGeometryKernelEPKNS0_13VPlacedVolumeE' cannot be statically determined
     863 nvlink info    : Function '_ZNK7vecgeom4cuda31ScalarShapeImplementationHelperINS0_22PolyconeImplementationILin1ELin1EEEE8ContainsERKNS0_8Vector3DIdEE' has address taken but no possible call to it
     864 nvlink info    : Function '_ZNK7vecgeom4cuda31ScalarShapeImplementationHelperINS0_22PolyconeImplementationILin1ELin1EEEE8ContainsERKNS0_8Vector3DIdEERS6_' has address taken but no possible call to it
     865 nvlink info    : Function '_ZNK7vecgeom4cuda31ScalarShapeImplementationHelperINS0_22PolyconeImplementationILin1ELin1EEEE16UnplacedContainsERKNS0_8Vector3DIdEE' has address taken but no possible call to it
    ....
    1297 nvlink info    : Function '_ZNK7vecgeom4cuda31ScalarShapeImplementationHelperINS0_21BooleanImplementationIL16BooleanOperation2ELin1ELin1EEEE13DistanceToOutERKNS0_8Vector3DIdEES9_d' has address taken but no possible call to it
    1298 nvlink info    : Function '_ZNK7vecgeom4cuda31ScalarShapeImplementationHelperINS0_21BooleanImplementationIL16BooleanOperation2ELin1ELin1EEEE19PlacedDistanceToOutERKNS0_8Vector3DIdEES9_d' has address taken but no possible call to it
    1299 nvlink info    : Function '_ZNK7vecgeom4cuda31ScalarShapeImplementationHelperINS0_21BooleanImplementationIL16BooleanOperation2ELin1ELin1EEEE10SafetyToInERKNS0_8Vector3DIdEE' has address taken but no possible call to it
    1300 nvlink info    : Function '_ZNK7vecgeom4cuda31ScalarShapeImplementationHelperINS0_21BooleanImplementationIL16BooleanOperation2ELin1ELin1EEEE11SafetyToOutERKNS0_8Vector3DIdEE' has address taken but no possible call to it
    1301 Scanning dependencies of target cudauserlib
    1302 [ 87%] Linking CXX shared library libcudauserlib.so
    1303 [ 87%] Built target cudauserlib
    ...
    1634 -- Installing: /usr/local/env/geometry/VecGeom.install/include/backend/umesimd/Backend.h
    1635 -- Installing: /usr/local/env/geometry/VecGeom.install/include/backend/vc
    1636 -- Installing: /usr/local/env/geometry/VecGeom.install/include/backend/vc/Backend.h
    1637 -- Installing: /usr/local/env/geometry/VecGeom.install/lib/libvecgeom.a
    1638 -- Installing: /usr/local/env/geometry/VecGeom.install/lib/libvecgeomcuda.so
    1639 -- Installing: /usr/local/env/geometry/VecGeom.install/lib/libvecgeomcuda_static.a
    1640 -- Installing: /usr/local/env/geometry/VecGeom.install/include/VecCore/Config.h
    1641 -- Up-to-date: /usr/local/env/geometry/VecGeom.install/include/VecCore
    1642 -- Installing: /usr/local/env/geometry/VecGeom.install/include/VecCore/Assert.h


::

    simon:VecGeom.build blyth$ vecgeom-t
    (lldb) target create "./OrbBenchmark"
    Current executable set to './OrbBenchmark' (x86_64).
    (lldb) r
    Process 37696 launched: './OrbBenchmark' (x86_64)
    INFO: using default 10240 for option -npoints
    INFO: using default 1 for option -nrep
    INFO: using default 3 for option -r
    PlacedVolume created after geometry is closed --> will not be registered
    PlacedVolume created after geometry is closed --> will not be registered
    Running Contains and Inside benchmark for 10240 points for 1 repetitions.
    Generating points with bias 0.500000... Done in 0.008024 s.
    Vectorized    - Inside: 0.001403s (0.001403s), Contains: 0.001313s (0.001313s), Inside/Contains: 1.07
    Specialized   - Inside: 0.001246s (0.001246s), Contains: 0.001184s (0.001184s), Inside/Contains: 1.05
    Unspecialized - Inside: 0.001252s (0.001252s), Contains: 0.001198s (0.001198s), Inside/Contains: 1.05
    CUDA          - ScanGeometry found pvolumes2
    Starting synchronization to GPU.
    Allocating geometry on GPU...Allocating logical volumes... OK
    Allocating unplaced volumes... OK
    Allocating placed volume Assertion failed: (vpv != nullptr), function AllocatePlacedVolumesOnCoproc, file /usr/local/env/geometry/VecGeom/source/CudaManager.cpp, line 259.
    Process 37696 stopped
    * thread #1: tid = 0x288414, 0x00007fff9643e866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff9643e866 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill + 10:
    -> 0x7fff9643e866:  jae    0x7fff9643e870            ; __pthread_kill + 20
       0x7fff9643e868:  movq   %rax, %rdi
       0x7fff9643e86b:  jmp    0x7fff9643b175            ; cerror_nocancel
       0x7fff9643e870:  retq   
    (lldb) bt
    * thread #1: tid = 0x288414, 0x00007fff9643e866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff9643e866 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff8dadb35c libsystem_pthread.dylib`pthread_kill + 92
        frame #2: 0x00007fff9482bb1a libsystem_c.dylib`abort + 125
        frame #3: 0x00007fff947f59bf libsystem_c.dylib`__assert_rtn + 321
        frame #4: 0x0000000104f0f96b libvecgeomcuda.so`vecgeom::cxx::CudaManager::AllocatePlacedVolumesOnCoproc(this=0x00000001000a1d98) + 283 at CudaManager.cpp:259
        frame #5: 0x0000000104f0d890 libvecgeomcuda.so`vecgeom::cxx::CudaManager::AllocateGeometry(this=0x00000001000a1d98) + 1424 at CudaManager.cpp:315
        frame #6: 0x0000000104f0b167 libvecgeomcuda.so`vecgeom::cxx::CudaManager::Synchronize(this=0x00000001000a1d98) + 183 at CudaManager.cpp:66
        frame #7: 0x0000000104f00292 libvecgeomcuda.so`vecgeom::Benchmarker::GetVolumePointers(this=0x00007fff5fbfea08, volumesGpu=0x00007fff5fbfd178) + 98 at Benchmarker.cpp:2670
        frame #8: 0x0000000104eacdc9 libvecgeomcuda.so`vecgeom::Benchmarker::RunInsideCuda(this=0x00007fff5fbfea08, posX=0x000000010c130600, posY=0x000000010c144600, posZ=0x000000010c158600, contains=0x000000010c18ce00, inside=0x000000010c18f600) + 329 at Benchmarker.cu:78
        frame #9: 0x000000010004ef5e OrbBenchmark`vecgeom::Benchmarker::RunInsideBenchmark(this=0x00007fff5fbfea08) + 3806 at Benchmarker.cpp:723
        frame #10: 0x000000010004e008 OrbBenchmark`vecgeom::Benchmarker::RunBenchmark(this=0x00007fff5fbfea08) + 104 at Benchmarker.cpp:623
        frame #11: 0x00000001000030c9 OrbBenchmark`main(argc=1, argv=0x00007fff5fbfedb0) + 1257 at OrbBenchmark.cpp:45
        frame #12: 0x00007fff918b15fd libdyld.dylib`start + 1
    (lldb) 




EOU
}
vecgeom-edir(){ echo $(env-home)/geometry/vecgeom ; }
vecgeom-sdir(){ echo $(local-base)/env/geometry/VecGeom ; }
vecgeom-bdir(){ echo $(vecgeom-dir).build ; }
vecgeom-idir(){ echo $(vecgeom-dir).install ; }
vecgeom-dir(){  echo $(vecgeom-sdir) ; }

vecgeom-cd(){   cd $(vecgeom-dir); }
vecgeom-ecd(){  cd $(vecgeom-edir); }
vecgeom-scd(){  cd $(vecgeom-sdir); }
vecgeom-bcd(){  cd $(vecgeom-bdir); }
vecgeom-icd(){  cd $(vecgeom-idir); }

vecgeom-url(){ echo https://gitlab.cern.ch/VecGeom/VecGeom.git ; }
vecgeom-name(){ echo VecGeom ; }
vecgeom-get(){
   local dir=$(dirname $(vecgeom-dir)) &&  mkdir -p $dir && cd $dir
   [ ! -d "$(vecgeom-name)" ] && git clone $(vecgeom-url) 
}

vecgeom-env(){      elocal- ; cuda- ; }



vecgeom-find(){ 
   local q=${1:-RunInsideCuda}
   vecgeom-scd
   find . -type f -exec grep -H $q {} \;
}

vecgeom-cmake(){
   local iwd=$PWD
   local bdir=$(vecgeom-bdir)

   [ ! -d "$bdir" ] && mkdir -p $bdir

   vecgeom-bcd

   cmake $(vecgeom-sdir) \
          -DCMAKE_BUILD_TYPE=Debug \
          -DCMAKE_INSTALL_PREFIX=$(vecgeom-idir) \
          -DCUDA=ON \
          -DCUDA_ARCH=30 \
          -DBENCHMARK=ON \
          $*

   cd $iwd
}

vecgeom--()
{
   vecgeom-bcd
   make install
}


vecgeom-t()
{
    vecgeom-bcd

    lldb ./OrbBenchmark


# (lldb) b GeoManager::RegisterPlacedVolume


}

vecgeom-nvcc()
{
    /usr/local/cuda/bin/nvcc \
          /usr/local/env/geometry/VecGeom/source/CudaManager_0.cu \
           -dc \
             -o /usr/local/env/geometry/VecGeom.build/CMakeFiles/vecgeomcuda.dir/source/./vecgeomcuda_generated_CudaManager.cu.o \
             -ccbin /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang \
              -m64 \
              -Dvecgeomcuda_EXPORTS \
              -Xcompiler ,\"-g\",\"-stdlib=libc++\",\"-Wall\",\"-fPIC\",\"-ggdb\",\"-O0\",\"-march=native\",\"-DVECGEOM_SCALAR\",\"-DVECGEOM_CUDA\",\"-DVECGEOM_CUDA_NO_VOLUME_SPECIALIZATION\",\"-DVECGEOM_QUADRILATERALS_VC\",\"-DVECGEOM_NO_SPECIALIZATION\",\"-DVECGEOM_INPLACE_TRANSFORMATIONS\",\"-DVECGEOM_USE_INDEXEDNAVSTATES\",\"-march=native\",\"-fPIC\",\"-g\" \
               -std=c++11 \
               -Xcompiler \
               -Wno-unused-function \
               -Xcudafe \
               --diag_suppress=code_is_unreachable \
               -Xcudafe \
               --diag_suppress=initialization_not_reachable \
               -arch=sm_30 \
               -g \
               -G \
               -DNVCC \
               -I/usr/local/cuda/include \
               -I/usr/local/env/geometry/VecGeom \
               -I/usr/local/env/geometry/VecGeom/VecCore/include \
               -I/usr/local/env/geometry/VecGeom.build/VecCore/include \
               -I/usr/local/cuda/include

}


