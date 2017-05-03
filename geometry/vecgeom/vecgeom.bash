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

No, not practically for non-trivial shapes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OptiX intersection inputs are PTX files compiled from 
CUDA programs with specific signatures which get OptiX
compiled into a mega kernel. There is no possibility 
to link to other objects from a separate library. 
The only way that VecGeom could be used is at source level, 
ie if it had some real simple headers in which the intersection 
algorithms are provided. That would only be possible for the 
simplest shapes anyhow, that already have implementations of already.
More complex shapes need data structure workings that VecGeom
would use C++ constructs for, involving many files.

So can use VecGeom for algorithmic inspiration, not direct usage.


If they could be
~~~~~~~~~~~~~~~~~~~

* could avoid manual step of writing analytic geometry code
* BUT it is not a panacea, often the transition from volume to surface based geometry
  needed some cleanups of things like touching volumes that cause coincident surfaces
  which have to be fixed
* highly non-trivial, 


Build Issues
--------------


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




Working
---------

::

   test/cuda/MapTest.cpp testing base/Map.h (stl map adapted for CUDA)

All tests have rc 0::

    simon:VecGeom.build blyth$ l *Test
    -rwxr-xr-x  1 blyth  staff    53740 Jan 19 15:33 Transformation3DTest
    -rwxr-xr-x  1 blyth  staff  1058748 Jan 19 15:33 SafetyEstimatorTest
    -rwxr-xr-x  1 blyth  staff    47108 Jan 19 15:33 ThetaConeTest
    -rwxr-xr-x  1 blyth  staff    54084 Jan 19 15:33 MapTest
    -rwxr-xr-x  1 blyth  staff    31524 Jan 19 15:33 PhiWedgeTest
    -rwxr-xr-x  1 blyth  staff  1062392 Jan 19 15:33 PlanesTest
    -rwxr-xr-x  1 blyth  staff     9336 Jan 19 15:33 QuadrilateralTest
    -rwxr-xr-x  1 blyth  staff  1091500 Jan 19 15:33 AssemblyTest
    -rwxr-xr-x  1 blyth  staff    36892 Jan 19 15:33 BitSetTest
    -rwxr-xr-x  1 blyth  staff  1205852 Jan 19 15:33 BooleanConvexityTest
    -rwxr-xr-x  1 blyth  staff    31140 Jan 19 15:33 ContainerTest



ctests
-------

All benchmark fail::

    simon:VecGeom.build blyth$ make test
    Running tests...
    Test project /usr/local/env/geometry/VecGeom.build
          Start  1: SafetyEstimatorTest
     1/42 Test  #1: SafetyEstimatorTest ..............   Passed    0.00 sec
          Start  2: ContainerTest
     2/42 Test  #2: ContainerTest ....................   Passed    0.00 sec
          Start  3: create_geometry
     3/42 Test  #3: create_geometry ..................   Passed    0.01 sec
          Start  4: PlanesTest
     4/42 Test  #4: PlanesTest .......................   Passed    0.01 sec
          Start  5: QuadrilateralTest
     5/42 Test  #5: QuadrilateralTest ................   Passed    0.00 sec
          Start  6: Transformation3DTest
     6/42 Test  #6: Transformation3DTest .............   Passed    0.00 sec
          Start  7: PhiWedgeTest
     7/42 Test  #7: PhiWedgeTest .....................   Passed    0.00 sec
          Start  8: ThetaConeTest
     8/42 Test  #8: ThetaConeTest ....................   Passed    0.00 sec
          Start  9: TestConvexity
     9/42 Test  #9: TestConvexity ....................   Passed    0.02 sec
          Start 10: BooleanConvexityTest
    10/42 Test #10: BooleanConvexityTest .............   Passed    0.00 sec
          Start 11: TestVecGeomPolycone
    11/42 Test #11: TestVecGeomPolycone ..............   Passed    0.01 sec
          Start 12: TestSExtru
    12/42 Test #12: TestSExtru .......................   Passed    0.25 sec
          Start 13: TestBooleans
    13/42 Test #13: TestBooleans .....................   Passed    0.00 sec
          Start 14: AssemblyTest
    14/42 Test #14: AssemblyTest .....................   Passed    0.00 sec
          Start 15: TestNavigationState
    15/42 Test #15: TestNavigationState ..............   Passed    0.01 sec
          Start 16: BitSetTest
    16/42 Test #16: BitSetTest .......................   Passed    0.00 sec
          Start 17: BoxBenchmark
    17/42 Test #17: BoxBenchmark .....................***Exception: Other  0.58 sec
          Start 18: SExtruBenchmark
    18/42 Test #18: SExtruBenchmark ..................***Exception: Other  0.56 sec
          Start 19: ConcaveSExtruBenchmark
    19/42 Test #19: ConcaveSExtruBenchmark ...........***Exception: Other  0.57 sec
          Start 20: ParaboloidBenchmark
    20/42 Test #20: ParaboloidBenchmark ..............***Exception: Other  0.52 sec
          Start 21: ParaboloidScriptBenchmark
    21/42 Test #21: ParaboloidScriptBenchmark ........***Exception: Other  1.24 sec
          Start 22: ParallelepipedBenchmark
    22/42 Test #22: ParallelepipedBenchmark ..........***Exception: Other  0.59 sec
          Start 23: PolyhedronBenchmark
    23/42 Test #23: PolyhedronBenchmark ..............***Exception: Other  0.58 sec
          Start 24: TubeBenchmark
    24/42 Test #24: TubeBenchmark ....................***Exception: Other  0.56 sec
          Start 25: TorusBenchmark2
    25/42 Test #25: TorusBenchmark2 ..................***Exception: Other  0.54 sec
          Start 26: TrapezoidBenchmark
    26/42 Test #26: TrapezoidBenchmark ...............***Exception: Other  0.69 sec
          Start 27: TrapezoidBenchmarkScript
    27/42 Test #27: TrapezoidBenchmarkScript .........***Exception: Other  1.55 sec
          Start 28: OrbBenchmark
    28/42 Test #28: OrbBenchmark .....................***Exception: Other  0.53 sec
          Start 29: SphereBenchmark
    29/42 Test #29: SphereBenchmark ..................***Exception: Other  0.48 sec
          Start 30: HypeBenchmark
    30/42 Test #30: HypeBenchmark ....................***Exception: Other  0.53 sec
          Start 31: TrdBenchmark
    31/42 Test #31: TrdBenchmark .....................***Exception: Other  0.61 sec
          Start 32: ConeBenchmark
    32/42 Test #32: ConeBenchmark ....................***Exception: Other  0.52 sec
          Start 33: GenTrapBenchmark
    33/42 Test #33: GenTrapBenchmark .................***Exception: Other  0.52 sec
          Start 34: ScaledBenchmark
    34/42 Test #34: ScaledBenchmark ..................***Exception: Other  0.66 sec
          Start 35: BoxScaledBenchmark
    35/42 Test #35: BoxScaledBenchmark ...............***Exception: Other  0.55 sec
          Start 36: CutTubeBenchmark
    36/42 Test #36: CutTubeBenchmark .................***Exception: Other  0.49 sec
          Start 37: PolyconeBenchmark
    37/42 Test #37: PolyconeBenchmark ................***Exception: Other  0.53 sec
          Start 38: Backend
    38/42 Test #38: Backend ..........................   Passed    0.02 sec
          Start 39: TypeTraits
    39/42 Test #39: TypeTraits .......................   Passed    0.01 sec
          Start 40: NumericLimits
    40/42 Test #40: NumericLimits ....................   Passed    0.00 sec
          Start 41: Math
    41/42 Test #41: Math .............................   Passed    0.01 sec
          Start 42: CUDAHelloWorld
    42/42 Test #42: CUDAHelloWorld ...................   Passed    0.12 sec

    50% tests passed, 21 tests failed out of 42

    Total Test time (real) =  13.90 sec

    The following tests FAILED:
         17 - BoxBenchmark (OTHER_FAULT)
         18 - SExtruBenchmark (OTHER_FAULT)
         19 - ConcaveSExtruBenchmark (OTHER_FAULT)
         20 - ParaboloidBenchmark (OTHER_FAULT)
         21 - ParaboloidScriptBenchmark (OTHER_FAULT)
         22 - ParallelepipedBenchmark (OTHER_FAULT)
         23 - PolyhedronBenchmark (OTHER_FAULT)
         24 - TubeBenchmark (OTHER_FAULT)
         25 - TorusBenchmark2 (OTHER_FAULT)
         26 - TrapezoidBenchmark (OTHER_FAULT)
         27 - TrapezoidBenchmarkScript (OTHER_FAULT)
         28 - OrbBenchmark (OTHER_FAULT)
         29 - SphereBenchmark (OTHER_FAULT)
         30 - HypeBenchmark (OTHER_FAULT)
         31 - TrdBenchmark (OTHER_FAULT)
         32 - ConeBenchmark (OTHER_FAULT)
         33 - GenTrapBenchmark (OTHER_FAULT)
         34 - ScaledBenchmark (OTHER_FAULT)
         35 - BoxScaledBenchmark (OTHER_FAULT)
         36 - CutTubeBenchmark (OTHER_FAULT)
         37 - PolyconeBenchmark (OTHER_FAULT)
    Errors while running CTest
    make: *** [test] Error 8



Many of the fails emit the below::

     36 void GeoManager::RegisterPlacedVolume(VPlacedVolume *const placed_volume)
     37 {
     38   if (!fIsClosed)
     39     fPlacedVolumesMap[placed_volume->id()] = placed_volume;
     40   else {
     41     std::cerr << "PlacedVolume created after geometry is closed --> will not be registered\n";
     42   }
     43 }

::

    simon:VecGeom blyth$ vecgeom-find RegisterPlacedVolume
    ./management/GeoManager.h:  void RegisterPlacedVolume(VPlacedVolume *const placed_volume);
    ./source/GeoManager.cpp:void GeoManager::RegisterPlacedVolume(VPlacedVolume *const placed_volume)
    ./source/PlacedVolume.cpp:  GeoManager::Instance().RegisterPlacedVolume(this);


UnplacedPolycone
-------------------

::

    simon:VecGeom blyth$ vecgeom-lfind UnplacedPolycone
    ./volumes/UnplacedPolycone.h
    ./source/UnplacedPolycone.cpp

::

    312 #ifdef VECGEOM_CUDA_INTERFACE
    313 
    314 DevicePtr<cuda::VUnplacedVolume> UnplacedPolycone::CopyToGpu() const
    315 {
    316   return CopyToGpuImpl<UnplacedPolycone>();
    317 }
    318 
    319 DevicePtr<cuda::VUnplacedVolume> UnplacedPolycone::CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const
    320 {
    321 
    322   // idea: reconstruct defining arrays: copy them to GPU; then construct the
    323   // UnplacedPolycon object from scratch
    324   // on the GPU
    325   std::vector<Precision> rmin, z, rmax;
    326   ReconstructSectionArrays(z, rmin, rmax);
    327 
    328   // somehow this does not work:
    329   //        Precision *z_gpu_ptr = AllocateOnGpu<Precision>( (z.size() +
    330   // rmin.size() + rmax.size())*sizeof(Precision) );
    331   //        Precision *rmin_gpu_ptr = z_gpu_ptr + sizeof(Precision)*z.size();
    332   //        Precision *rmax_gpu_ptr = rmin_gpu_ptr +
    333   // sizeof(Precision)*rmin.size();
    334 
    335   Precision *z_gpu_ptr    = AllocateOnGpu<Precision>(z.size() * sizeof(Precision));
    336   Precision *rmin_gpu_ptr = AllocateOnGpu<Precision>(rmin.size() * sizeof(Precision));
    337   Precision *rmax_gpu_ptr = AllocateOnGpu<Precision>(rmax.size() * sizeof(Precision));
    338 
    339   vecgeom::CopyToGpu(&z[0], z_gpu_ptr, sizeof(Precision) * z.size());
    340   vecgeom::CopyToGpu(&rmin[0], rmin_gpu_ptr, sizeof(Precision) * rmin.size());
    341   vecgeom::CopyToGpu(&rmax[0], rmax_gpu_ptr, sizeof(Precision) * rmax.size());


Observations

* a very chatty approach, the Opticks way would be to build a buffer CPU side 
  and copy to GPU in one go

* does far more than is needed to just ray-primitive intersect


G4VCSGfaceted
----------------

Polycone implemented as collection of G4VCSGface::

    simon:solids blyth$ g4-cls G4VCSGfaceted
    vi -R source/geometry/solids/specific/include/G4VCSGfaceted.hh source/geometry/solids/specific/src/G4VCSGfaceted.cc
    2 files to edit

    simon:solids blyth$ g4-cls G4VCSGface
    vi -R source/geometry/solids/specific/include/G4VCSGface.hh

    simon:solids blyth$ g4-hh G4VCSGfaceted
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/geometry/solids/specific/include/G4GenericPolycone.hh:#include "G4VCSGfaceted.hh"
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/geometry/solids/specific/include/G4GenericPolycone.hh:class G4GenericPolycone : public G4VCSGfaceted 
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/geometry/solids/specific/include/G4Polycone.hh://   inherited from  class G4VCSGfaceted:
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/geometry/solids/specific/include/G4Polycone.hh:#include "G4VCSGfaceted.hh"
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/geometry/solids/specific/include/G4Polycone.hh:class G4Polycone : public G4VCSGfaceted 
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/geometry/solids/specific/include/G4Polyhedra.hh://   inherited from class G4VCSGfaceted:
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/geometry/solids/specific/include/G4Polyhedra.hh:#include "G4VCSGfaceted.hh"
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/geometry/solids/specific/include/G4Polyhedra.hh:class G4Polyhedra : public G4VCSGfaceted
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/geometry/solids/specific/include/G4VCSGface.hh://   In analogy with CalculateExtent for G4VCSGfaceted, this is

Polycone with conical faces joining RZ::

    102 class G4PolyconeSide : public G4VCSGface
    103 {
    104   public:
    105 
    106     G4PolyconeSide( const G4PolyconeSideRZ *prevRZ,
    107                     const G4PolyconeSideRZ *tail,
    108                     const G4PolyconeSideRZ *head,
    109                     const G4PolyconeSideRZ *nextRZ,
    110                           G4double phiStart, G4double deltaPhi,
    111                           G4bool phiIsOpen, G4bool isAllBehind=false );
    ...
    117     G4bool Intersect( const G4ThreeVector &p, const G4ThreeVector &v,
    118                             G4bool outgoing, G4double surfTolerance,
    119                             G4double &distance, G4double &distFromSurface,
    120                             G4ThreeVector &normal, G4bool &isAllBehind );
    121 
    122     G4double Distance( const G4ThreeVector &p, G4bool outgoing );



Created in G4Polycone.cc::

     277   // Construct conical faces
     ...
     281   G4PolyconeSideRZ *corner = corners,
     282                    *prev = corners + numCorner-1,
     283                    *nextNext;
     284   G4VCSGface  **face = faces;
     285   do    // Loop checking, 13.08.2015, G.Cosmo
     286   {
     287     next = corner+1;
     288     if (next >= corners+numCorner) next = corners;
     289     nextNext = next+1;
     290     if (nextNext >= corners+numCorner) nextNext = corners;
     291    
     292     if (corner->r < 1/kInfinity && next->r < 1/kInfinity) continue;
     ...
     315     *face++ = new G4PolyconeSide( prev, corner, next, nextNext,
     316                 startPhi, endPhi-startPhi, phiIsOpen, allBehind );
     317   } while( prev=corner, corner=next, corner > corners );


::

     073 // Values for r1,z1 and r2,z2 should be specified in clockwise
      74 // order in (r,z).
      75 //
      76 G4PolyconeSide::G4PolyconeSide( const G4PolyconeSideRZ *prevRZ,
      77                                 const G4PolyconeSideRZ *tail,
      78                                 const G4PolyconeSideRZ *head,
      79                                 const G4PolyconeSideRZ *nextRZ,
      80                                       G4double thePhiStart,
      81                                       G4double theDeltaPhi,
      82                                       G4bool thePhiIsOpen,
      83                                       G4bool isAllBehind )
      84   : ncorners(0), corners(0)
      85 {
      ..
      94   //
      95   // Record values
      96   //
      97   r[0] = tail->r; z[0] = tail->z;
      98   r[1] = head->r; z[1] = head->z;
      99  
     138   // Make our intersecting cone
     139   //
     140   cone = new G4IntersectingCone( r, z );
     141 



Itersect called for all the faces, min distance one returned::

    264 // DistanceToIn(p,v)
    265 //
    266 G4double G4VCSGfaceted::DistanceToIn( const G4ThreeVector &p,
    267                                       const G4ThreeVector &v ) const
    268 {
    269   G4double distance = kInfinity;
    270   G4double distFromSurface = kInfinity;
    271   G4VCSGface **face = faces;
    272   G4VCSGface *bestFace = *face;
    273   do    // Loop checking, 13.08.2015, G.Cosmo
    274   {
    275     G4double   faceDistance,
    276                faceDistFromSurface;
    277     G4ThreeVector   faceNormal;
    278     G4bool    faceAllBehind;
    279     if ((*face)->Intersect( p, v, false, kCarTolerance/2,
    280                 faceDistance, faceDistFromSurface,
    281                 faceNormal, faceAllBehind ) )
    282     {
    283       //
    284       // Intersecting face
    285       //
    286       if (faceDistance < distance)
    287       {
    288         distance = faceDistance;
    289         distFromSurface = faceDistFromSurface;
    290         bestFace = *face;
    291         if (distFromSurface <= 0) { return 0; }
    292       }
    293     }
    294   } while( ++face < faces + numFace );
    295 
    296   if (distance < kInfinity && distFromSurface<kCarTolerance/2)
    297   {
    298     if (bestFace->Distance(p,false) < kCarTolerance/2)  { distance = 0; }
    299   }
    300 
    301   return distance;
    302 }






::

     g4-cls G4IntersectingCone  ## implements the actual intersection

     51 class G4IntersectingCone
     52 {
     53   public:
     54 
     55     G4IntersectingCone( const G4double r[2], const G4double z[2] );




G4BREPSolid
--------------

* https://indico.cern.ch/event/44566/contributions/1101956/attachments/943084/1337691/BREP_Solids.pps.pps
* ~/opticks_refs/BREP_Solids.pdf   

  * Gabriele Camellini (circa 2009) 
  * BREPS solids construction by surfaces of extrusion & revolution

  * G4BREPSolid defined by G4Surface and G4Curve (G4BREPSolid is defined by a collections of boundaried surfaces)
  * G4SurfaceOfLinearExtrusion 

* BREPS look to be expunged ? 

* https://gitlab.cern.ch/geant4/geant4/tree/edb408b5618b3b1cd3f40c5759aa5da4aa56bb7b/source/geometry/solids/BREPS/src


How come need so few methods needed  ? 
------------------------------------------------

* Dont need to know if are inside some volume ? the question never arises.

  * this is needed for converting CSG into meshed B-Rep 


For optical photons with boundary geometry the only questions are 

* whats the bbox of the primitive

* does a ray from here in that direction intersect with this primitive
* if it does

  * whats the distance (parametric t) at 1st intersection
  * whats the geometric normal at the intersection

* can partition solids, when that is convenient (eg manual PMT partitioning)


Not Working
------------


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

::

    239 // a special treatment for placed volumes to ensure same order of placed volumes in compact buffer
    240 // as on CPU
    241 bool CudaManager::AllocatePlacedVolumesOnCoproc()
    242 {
    243   // check if geometry is closed
    244   if (!GeoManager::Instance().IsClosed()) {
    245     std::cerr << "Warning: Geometry on host side MUST be closed before copying to DEVICE\n";
    246   }
    247 
    248   // we start from the compact buffer on the CPU
    249   unsigned int size = placed_volumes_.size();
    250 
    251   //   if (verbose_ > 2) std::cout << "Allocating placed volume ";
    252   std::cerr << "Allocating placed volume ";
    253   size_t totalSize = 0;
    254   // calculate total size of buffer on GPU to hold the GPU copies of the collection
    255   for (unsigned int i = 0; i < size; ++i) {
    256 
    257     VPlacedVolume* const vpv = &GeoManager::gCompactPlacedVolBuffer[i] ;
    258     //assert(&GeoManager::gCompactPlacedVolBuffer[i] != nullptr);
    259     assert(vpv != nullptr);
    260     //totalSize += (&GeoManager::gCompactPlacedVolBuffer[i])->DeviceSizeOf();
    261     totalSize += vpv->DeviceSizeOf();
    262   }
    263 
    264   GpuAddress gpu_address;
    265   gpu_address.Allocate(totalSize);
    266 
    267   // store this address for later access (on the host)
    268   fPlacedVolumeBufferOnDevice = DevicePtr<vecgeom::cuda::VPlacedVolume>(gpu_address);
    269   // this address has to be made known globally to the device side
    270   vecgeom::cuda::InitDeviceCompactPlacedVolBufferPtr(gpu_address.GetPtr());
    271 
    272   allocated_memory_.push_back(gpu_address);
    273 
    274   // record a GPU memory location for each object in the collection to be copied
    275   // since the pointers in GeoManager::gCompactPlacedVolBuffer are sorted by the volume id, we are
    276   // getting the same order on the GPU/device automatically
    277   for (unsigned int i = 0; i < size; ++i) {
    278     VPlacedVolume const *ptr                  = &GeoManager::gCompactPlacedVolBuffer[i];
    279     memory_map[ToCpuAddress(ptr)]             = gpu_address;
    280     fGPUtoCPUmapForPlacedVolumes[gpu_address] = ptr;
    281     gpu_address += ptr->DeviceSizeOf();
    282   }
    283 
    284   if (verbose_ > 2) std::cout << " OK\n";
    285 
    286   return true;
    287 }



::

    (lldb) p i
    (unsigned int) $0 = 0
    (lldb) p GeoManager::gCompactPlacedVolBuffer
    (vecgeom::cxx::VPlacedVolume *) $1 = 0x000000010b63c620
    (lldb) p GeoManager::gCompactPlacedVolBuffer[0]
    (vecgeom::cxx::VPlacedVolume) $2 = {
      id_ = 0
      label_ = "p4r4"
      logical_volume_ = 0x00007fff5fbfebc0
      fTransformation = {
        fTranslation = ([0] = 5, [1] = 5, [2] = 5)
        fRotation = ([0] = 1, [1] = 0, [2] = 0, [3] = 0, [4] = 1, [5] = 0, [6] = 0, [7] = 0, [8] = 1)
        fIdentity = false
        fHasRotation = false
        fHasTranslation = true
      }
      bounding_box_ = 0x0000000000000000
    }
    (lldb) p &GeoManager::gCompactPlacedVolBuffer[0]
    (vecgeom::cxx::VPlacedVolume *) $3 = 0x000000010b63c620
    (lldb) p vpv
    (vecgeom::cxx::VPlacedVolume *const) $4 = 0x0000000000000000
    (lldb) p *&GeoManager::gCompactPlacedVolBuffer[0]
    (vecgeom::cxx::VPlacedVolume) $5 = {
      id_ = 0
      label_ = "p4r4"
      logical_volume_ = 0x00007fff5fbfebc0
      fTransformation = {
        fTranslation = ([0] = 5, [1] = 5, [2] = 5)
        fRotation = ([0] = 1, [1] = 0, [2] = 0, [3] = 0, [4] = 1, [5] = 0, [6] = 0, [7] = 0, [8] = 1)
        fIdentity = false
        fHasRotation = false
        fHasTranslation = true
      }
      bounding_box_ = 0x0000000000000000
    }
    (lldb) expr VPlacedVolume* const vpv0=&GeoManager::gCompactPlacedVolBuffer[i]
    error: reference to 'VPlacedVolume' is ambiguous
    note: candidate found by name lookup is 'vecgeom::cxx::VPlacedVolume'
    note: candidate found by name lookup is 'vecgeom::cxx::VPlacedVolume'
    error: 1 errors parsing expression
    (lldb) expr vecgeom::cxx::VPlacedVolume* const vpv0 = &GeoManager::gCompactPlacedVolBuffer[i]
    error: reference to 'VPlacedVolume' is ambiguous
    note: candidate found by name lookup is 'vecgeom::cxx::VPlacedVolume'
    note: candidate found by name lookup is 'vecgeom::cxx::VPlacedVolume'
    error: 1 errors parsing expression

    (lldb) p *(GeoManager::gCompactPlacedVolBuffer + i)
    (vecgeom::cxx::VPlacedVolume) $8 = {
      id_ = 0
      label_ = "p4r4"
      logical_volume_ = 0x00007fff5fbfebc0
      fTransformation = {
        fTranslation = ([0] = 5, [1] = 5, [2] = 5)
        fRotation = ([0] = 1, [1] = 0, [2] = 0, [3] = 0, [4] = 1, [5] = 0, [6] = 0, [7] = 0, [8] = 1)
        fIdentity = false
        fHasRotation = false
        fHasTranslation = true
      }
      bounding_box_ = 0x0000000000000000
    }


::

    simon:opticks blyth$ vecgeom-find gCompactPlacedVolBuffer
    ./management/CudaManager.h:// extern __device__ VPlacedVolume *gCompactPlacedVolBuffer;
    ./management/GeoManager.h:  static VPlacedVolume *gCompactPlacedVolBuffer;
    ./navigation/NavigationState.h:    return &vecgeom::GeoManager::gCompactPlacedVolBuffer[index];
    ./navigation/NavigationState.h:    // (failed preveously due to undefined symbol vecgeom::cuda::GeoManager::gCompactPlacedVolBuffer)
    ./services/NavigationSpecializer.cpp:    outstream << "VPlacedVolume const * pvol = &GeoManager::gCompactPlacedVolBuffer[" << fTargetVolIds[transitionid]
    ./source/CudaGlobalSymbols.cu:__device__ VPlacedVolume *gCompactPlacedVolBuffer;
    ./source/CudaManager.cpp:           (size_t)(&GeoManager::gCompactPlacedVolBuffer[0]) + sizeof(vecgeom::cxx::VPlacedVolume) * (*i)->id());
    ./source/CudaManager.cpp:    //VPlacedVolume* const vpv = &GeoManager::gCompactPlacedVolBuffer[i] ; 
    ./source/CudaManager.cpp:    //assert(&GeoManager::gCompactPlacedVolBuffer[i] != nullptr);
    ./source/CudaManager.cpp:    VPlacedVolume* vpv = GeoManager::gCompactPlacedVolBuffer + i ; 
    ./source/CudaManager.cpp:    //totalSize += (&GeoManager::gCompactPlacedVolBuffer[i])->DeviceSizeOf();
    ./source/CudaManager.cpp:  // since the pointers in GeoManager::gCompactPlacedVolBuffer are sorted by the volume id, we are
    ./source/CudaManager.cpp:    VPlacedVolume const *ptr                  = &GeoManager::gCompactPlacedVolBuffer[i];
    ./source/CudaManager.cu:static __device__ VPlacedVolume *gCompactPlacedVolBuffer = nullptr;
    ./source/CudaManager.cu:  return gCompactPlacedVolBuffer;
    ./source/CudaManager_0.cu:static __device__ VPlacedVolume *gCompactPlacedVolBuffer = nullptr;
    ./source/CudaManager_0.cu:  return gCompactPlacedVolBuffer;
    ./source/GeoManager.cpp:VPlacedVolume *GeoManager::gCompactPlacedVolBuffer = nullptr;
    ./source/GeoManager.cpp:  gCompactPlacedVolBuffer = (VPlacedVolume *)malloc(pvolumecount * sizeof(VPlacedVolume));
    ./source/GeoManager.cpp:    gCompactPlacedVolBuffer[volumeindex] = *v.second;
    ./source/GeoManager.cpp:    fPlacedVolumesMap[volumeindex]       = &gCompactPlacedVolBuffer[volumeindex];
    ./source/GeoManager.cpp:    conversionmap[v.second]              = &gCompactPlacedVolBuffer[volumeindex];
    ./source/GeoManager.cpp:  if (GeoManager::gCompactPlacedVolBuffer != nullptr) {
    ./source/GeoManager.cpp:    free(gCompactPlacedVolBuffer);
    ./source/GeoManager.cpp:    gCompactPlacedVolBuffer = nullptr;
    simon:VecGeom blyth$ 

source/CudaGlobalSymbols.cu::

     01 #include "base/Global.h"
      2 
      3 namespace vecgeom {
      4 class VPlacedVolume;
      5 // instantiation of global device geometry data
      6 namespace globaldevicegeomdata {
      7 //#ifdef VECGEOM_NVCC_DEVICE
      8 __device__ VPlacedVolume *gCompactPlacedVolBuffer;
      9 //#endif
     10 }
     11 }





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
    local q="${1:-RunInsideCuda}"
    vecgeom-scd
    find . -type f -exec grep -H "$q" {} \;
}

vecgeom-cls(){ 
    local iwd=$PWD;
    local name=${1:-CudaManager}
    vecgeom-scd
    local h=$(find . -name "$name.h");
    local hh=$(find . -name "$name.hh");
    local cc=$(find . -name "$name.cc");
    local cpp=$(find . -name "$name.cpp");
    local cu=$(find . -name "$name.cu");
    local vcmd="vi -R $h $hh $cc $cpp $cu";
    echo $vcmd;
    eval $vcmd;
    cd $iwd
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




