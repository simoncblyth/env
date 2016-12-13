# === func-gen- : geometry/vecgeom/vecgeom fgp geometry/vecgeom/vecgeom.bash fgn vecgeom fgh geometry/vecgeom
vecgeom-src(){      echo geometry/vecgeom/vecgeom.bash ; }
vecgeom-source(){   echo ${BASH_SOURCE:-$(env-home)/$(vecgeom-src)} ; }
vecgeom-vi(){       vi $(vecgeom-source) ; }
vecgeom-env(){      elocal- ; }
vecgeom-usage(){ cat << EOU

VecGeom : The vectorized geometry library for particle-detector simulation (toolkits).
========================================================================================

Sandro Christian Wenzel

* https://gitlab.cern.ch/VecGeom/VecGeom

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


Issues are hidden behind cern SSO which fails::

    Microsoft.IdentityServer.Web.RequestFailedException: MSIS7000: The sign in request is not compliant to the WS-Federation language for web browser clients or the SAML 2.0 protocol WebSSO profile.
    Reference number: e3d0ecc5-2ea2-4fea-befa-dba4121ddb3f



Intrinsic header compilation error
--------------------------------------

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


Benchmark asserts 
-------------------

::

    simon:VecGeom.build blyth$ ./TubeBenchmark
    dyld: Library not loaded: @rpath/libcudart.7.0.dylib
      Referenced from: /usr/local/env/geometry/VecGeom.build/./TubeBenchmark
      Reason: image not found
    Trace/BPT trap: 5
    simon:VecGeom.build blyth$ 
    simon:VecGeom.build blyth$ cuda-

    simon:VecGeom.build blyth$ lldb ./TubeBenchmark 
    (lldb) target create "./TubeBenchmark"
    Current executable set to './TubeBenchmark' (x86_64).
    (lldb) r
    Process 22650 launched: './TubeBenchmark' (x86_64)
    INFO: using default 10240 for option -npoints
    INFO: using default 1 for option -nrep
    INFO: using default 0 for option -rmin
    INFO: using default 5 for option -rmax
    INFO: using default 10 for option -dz
    INFO: using default 0 for option -sphi
    INFO: using default 6.28319 for option -dphi
    PlacedVolume created after geometry is closed --> will not be registered
    PlacedVolume created after geometry is closed --> will not be registered
    Running Contains and Inside benchmark for 10240 points for 1 repetitions.
    Generating points with bias 0.500000... Done in 0.006851 s.
    Vectorized    - Inside: 0.001402s (0.001402s), Contains: 0.001406s (0.001406s), Inside/Contains: 1.00
    Specialized   - Inside: 0.001209s (0.001209s), Contains: 0.001192s (0.001192s), Inside/Contains: 1.01
    Unspecialized - Inside: 0.001211s (0.001211s), Contains: 0.001182s (0.001182s), Inside/Contains: 1.02
    CUDA          - ScanGeometry found pvolumes2
    Allocating placed volume Assertion failed: (&GeoManager::gCompactPlacedVolBuffer[i] != nullptr), function AllocatePlacedVolumesOnCoproc, file /usr/local/env/geometry/VecGeom/source/CudaManager.cpp, line 256.
    Process 22650 stopped
    * thread #1: tid = 0xf34e2, 0x00007fff8f97a866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff8f97a866 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill + 10:
    -> 0x7fff8f97a866:  jae    0x7fff8f97a870            ; __pthread_kill + 20
       0x7fff8f97a868:  movq   %rax, %rdi
       0x7fff8f97a86b:  jmp    0x7fff8f977175            ; cerror_nocancel
       0x7fff8f97a870:  retq   
    (lldb) bt
    * thread #1: tid = 0xf34e2, 0x00007fff8f97a866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff8f97a866 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff8701735c libsystem_pthread.dylib`pthread_kill + 92
        frame #2: 0x00007fff8dd67b1a libsystem_c.dylib`abort + 125
        frame #3: 0x00007fff8dd319bf libsystem_c.dylib`__assert_rtn + 321
        frame #4: 0x0000000104f1e945 libvecgeomcuda.so`vecgeom::cxx::CudaManager::AllocatePlacedVolumesOnCoproc(this=0x00000001000ac838) + 277 at CudaManager.cpp:256
        frame #5: 0x0000000104f1c870 libvecgeomcuda.so`vecgeom::cxx::CudaManager::AllocateGeometry(this=0x00000001000ac838) + 1424 at CudaManager.cpp:311
        frame #6: 0x0000000104f1a147 libvecgeomcuda.so`vecgeom::cxx::CudaManager::Synchronize(this=0x00000001000ac838) + 183 at CudaManager.cpp:66
        frame #7: 0x0000000104f0f272 libvecgeomcuda.so`vecgeom::Benchmarker::GetVolumePointers(this=0x00007fff5fbfe630, volumesGpu=0x00007fff5fbfce48) + 98 at Benchmarker.cpp:2670
        frame #8: 0x0000000104ebbda9 libvecgeomcuda.so`vecgeom::Benchmarker::RunInsideCuda(this=0x00007fff5fbfe630, posX=0x000000010b805c00, posY=0x000000010b819c00, posZ=0x000000010b82dc00, contains=0x000000010b867a00, inside=0x000000010b86a200) + 329 at Benchmarker.cu:78
        frame #9: 0x000000010005580e TubeBenchmark`vecgeom::Benchmarker::RunInsideBenchmark(this=0x00007fff5fbfe630) + 3806 at Benchmarker.cpp:723
        frame #10: 0x00000001000548b8 TubeBenchmark`vecgeom::Benchmarker::RunBenchmark(this=0x00007fff5fbfe630) + 104 at Benchmarker.cpp:623
        frame #11: 0x00000001000027bb TubeBenchmark`benchmark(rmin=0, rmax=5, dz=10, sphi=0, dphi=6.2831853071795862, npoints=10240, nrep=1) + 523 at TubeBenchmark.cpp:29
        frame #12: 0x0000000100003013 TubeBenchmark`main(argc=1, argv=0x00007fff5fbfee18) + 1827 at TubeBenchmark.cpp:42
        frame #13: 0x00007fff8aded5fd libdyld.dylib`start + 1
        frame #14: 0x00007fff8aded5fd libdyld.dylib`start + 1
    (lldb) 

    (lldb) f 10
    frame #10: 0x00000001000548b8 TubeBenchmark`vecgeom::Benchmarker::RunBenchmark(this=0x00007fff5fbfe630) + 104 at Benchmarker.cpp:623
       620  {
       621    assert(fWorld != nullptr);
       622    int errorcode = 0;
    -> 623    errorcode += RunInsideBenchmark();
       624    errorcode += RunToInBenchmark();
       625    errorcode += RunToOutBenchmark();
       626    if (fMeasurementCount == 1) errorcode += CompareMetaInformation();

    (lldb) f 9
    frame #9: 0x000000010005580e TubeBenchmark`vecgeom::Benchmarker::RunInsideBenchmark(this=0x00007fff5fbfe630) + 3806 at Benchmarker.cpp:723
       720      if (fOkToRunROOT) RunInsideRoot(containsRoot);
       721  #endif
       722  #ifdef VECGEOM_CUDA
    -> 723      RunInsideCuda(fPointPool->x(), fPointPool->y(), fPointPool->z(), containsCuda, insideCuda);
       724  #endif
       725    }

    (lldb) f 8
    frame #8: 0x0000000104ebbda9 libvecgeomcuda.so`vecgeom::Benchmarker::RunInsideCuda(this=0x00007fff5fbfe630, posX=0x000000010b805c00, posY=0x000000010b819c00, posZ=0x000000010b82dc00, contains=0x000000010b867a00, inside=0x000000010b86a200) + 329 at Benchmarker.cu:78
       75     if (fVerbosity > 0) printf("CUDA          - ");
       76   
       77     std::list<CudaVolume> volumesGpu;
    -> 78     GetVolumePointers(volumesGpu);
       79   
       80     vecgeom::cuda::LaunchParameters launch = vecgeom::cuda::LaunchParameters(fPointCount);

    (lldb) f 7
    frame #7: 0x0000000104f0f272 libvecgeomcuda.so`vecgeom::Benchmarker::GetVolumePointers(this=0x00007fff5fbfe630, volumesGpu=0x00007fff5fbfce48) + 98 at Benchmarker.cpp:2670
       2667 void Benchmarker::GetVolumePointers(std::list<DevicePtr<cuda::VPlacedVolume>> &volumesGpu)
       2668 {
       2669   CudaManager::Instance().LoadGeometry(GetWorld());
    -> 2670   CudaManager::Instance().Synchronize();
       2671   for (std::list<VolumePointers>::const_iterator v = fVolumes.begin(); v != fVolumes.end(); ++v) {
       2672     volumesGpu.push_back(CudaManager::Instance().LookupPlaced(v->Specialized()));
       2673   }

    (lldb) f 6
    frame #6: 0x0000000104f1a147 libvecgeomcuda.so`vecgeom::cxx::CudaManager::Synchronize(this=0x00000001000ac838) + 183 at CudaManager.cpp:66
       63   
       64     // Populate the memory map with GPU addresses
       65   
    -> 66     AllocateGeometry();
       67   
       68     // Create new objects with pointers adjusted to point to GPU memory, then
       69     // copy them to the allocated memory locations on the GPU.

    (lldb) f 5
    frame #5: 0x0000000104f1c870 libvecgeomcuda.so`vecgeom::cxx::CudaManager::AllocateGeometry(this=0x00000001000ac838) + 1424 at CudaManager.cpp:311
       308  
       309    // the allocation for placed volumes is a bit different (due to compact buffer treatment), so we call a specialized
       310    // function
    -> 311    AllocatePlacedVolumesOnCoproc(); // for placed volumes
       312  
       313    // this we should only do if not using inplace transformations
       314    AllocateCollectionOnCoproc("transformations", transformations_);

    (lldb) f 4
    frame #4: 0x0000000104f1e945 libvecgeomcuda.so`vecgeom::cxx::CudaManager::AllocatePlacedVolumesOnCoproc(this=0x00000001000ac838) + 277 at CudaManager.cpp:256
       253    size_t totalSize = 0;
       254    // calculate total size of buffer on GPU to hold the GPU copies of the collection
       255    for (unsigned int i = 0; i < size; ++i) {
    -> 256      assert(&GeoManager::gCompactPlacedVolBuffer[i] != nullptr);
       257      totalSize += (&GeoManager::gCompactPlacedVolBuffer[i])->DeviceSizeOf();
       258    }
       259  




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
    256     assert(&GeoManager::gCompactPlacedVolBuffer[i] != nullptr);
    257     totalSize += (&GeoManager::gCompactPlacedVolBuffer[i])->DeviceSizeOf();
    258   }
    259 


::

    simon:VecGeom.build blyth$ ./OrbBenchmark 
    INFO: using default 10240 for option -npoints
    INFO: using default 1 for option -nrep
    INFO: using default 3 for option -r
    PlacedVolume created after geometry is closed --> will not be registered
    PlacedVolume created after geometry is closed --> will not be registered
    Running Contains and Inside benchmark for 10240 points for 1 repetitions.
    Generating points with bias 0.500000... Done in 0.007703 s.
    Vectorized    - Inside: 0.001440s (0.001440s), Contains: 0.001325s (0.001325s), Inside/Contains: 1.09
    Specialized   - Inside: 0.001386s (0.001386s), Contains: 0.001216s (0.001216s), Inside/Contains: 1.14
    Unspecialized - Inside: 0.001507s (0.001507s), Contains: 0.001328s (0.001328s), Inside/Contains: 1.13
    CUDA          - ScanGeometry found pvolumes2
    Allocating placed volume Assertion failed: (&GeoManager::gCompactPlacedVolBuffer[i] != nullptr), function AllocatePlacedVolumesOnCoproc, file /usr/local/env/geometry/VecGeom/source/CudaManager.cpp, line 256.
    Abort trap: 6


Huh in debugger looks like the assert should be satisfied::

    (lldb) f 4
    frame #4: 0x0000000104f14945 libvecgeomcuda.so`vecgeom::cxx::CudaManager::AllocatePlacedVolumesOnCoproc(this=0x00000001000a8418) + 277 at CudaManager.cpp:256
       253    size_t totalSize = 0;
       254    // calculate total size of buffer on GPU to hold the GPU copies of the collection
       255    for (unsigned int i = 0; i < size; ++i) {
    -> 256      assert(&GeoManager::gCompactPlacedVolBuffer[i] != nullptr);
       257      totalSize += (&GeoManager::gCompactPlacedVolBuffer[i])->DeviceSizeOf();
       258    }
       259  
    (lldb) p i
    (unsigned int) $0 = 0
    (lldb) p GeoManager::gCompactPlacedVolBuffer
    (vecgeom::cxx::VPlacedVolume *) $1 = 0x000000010b6284b0
    (lldb) p GeoManager::gCompactPlacedVolBuffer[0]
    (vecgeom::cxx::VPlacedVolume) $2 = {
      id_ = 0
      label_ = "paraboloid"
      logical_volume_ = 0x00007fff5fbfec48
      fTransformation = {
        fTranslation = ([0] = 0, [1] = 0, [2] = 0)
        fRotation = ([0] = 1, [1] = 0, [2] = 0, [3] = 0, [4] = 1, [5] = 0, [6] = 0, [7] = 0, [8] = 1)
        fIdentity = true
        fHasRotation = false
        fHasTranslation = false
      }
      bounding_box_ = 0x0000000000000000
    }
    (lldb) p GeoManager::gCompactPlacedVolBuffer
    (vecgeom::cxx::VPlacedVolume *) $3 = 0x000000010b6284b0
    (lldb) p i
    (unsigned int) $4 = 0
    (lldb) p GeoManager::gCompactPlacedVolBuffer[i]
    (vecgeom::cxx::VPlacedVolume) $5 = {
      id_ = 0
      label_ = "paraboloid"
      logical_volume_ = 0x00007fff5fbfec48
      fTransformation = {
        fTranslation = ([0] = 0, [1] = 0, [2] = 0)
        fRotation = ([0] = 1, [1] = 0, [2] = 0, [3] = 0, [4] = 1, [5] = 0, [6] = 0, [7] = 0, [8] = 1)
        fIdentity = true
        fHasRotation = false
        fHasTranslation = false
      }
      bounding_box_ = 0x0000000000000000
    }
    (lldb) p &GeoManager::gCompactPlacedVolBuffer[i]
    (vecgeom::cxx::VPlacedVolume *) $6 = 0x000000010b6284b0
    (lldb) p nullptr
    (nullptr_t) $7 = 0x0000000000000000
    (lldb) p &GeoManager::gCompactPlacedVolBuffer[i] != nullptr
    (bool) $8 = true
    (lldb) 


Smth fishy::

    (lldb) p &GeoManager::gCompactPlacedVolBuffer[i]
    (vecgeom::cxx::VPlacedVolume *) $4 = 0x000000010b628540
    (lldb) p vpv
    (vecgeom::cxx::VPlacedVolume *) $5 = 0x0000000000000000
    (lldb) p size
    (unsigned int) $6 = 2

::

    simon:VecGeom.build blyth$ ./OrbBenchmark
    INFO: using default 10240 for option -npoints
    INFO: using default 1 for option -nrep
    INFO: using default 3 for option -r
    PlacedVolume created after geometry is closed --> will not be registered
    PlacedVolume created after geometry is closed --> will not be registered
    Running Contains and Inside benchmark for 10240 points for 1 repetitions.
    Generating points with bias 0.500000... Done in 0.008925 s.
    Vectorized    - Inside: 0.001493s (0.001493s), Contains: 0.001388s (0.001388s), Inside/Contains: 1.08
    Specialized   - Inside: 0.001301s (0.001301s), Contains: 0.001245s (0.001245s), Inside/Contains: 1.04
    Unspecialized - Inside: 0.001467s (0.001467s), Contains: 0.001243s (0.001243s), Inside/Contains: 1.18
    CUDA          - ScanGeometry found pvolumes2
    Starting synchronization to GPU.
    Allocating geometry on GPU...Allocating logical volumes... OK
    Allocating unplaced volumes... OK
    Allocating placed volume Assertion failed: (vpv != nullptr), function AllocatePlacedVolumesOnCoproc, file /usr/local/env/geometry/VecGeom/source/CudaManager.cpp, line 259.
    Abort trap: 6
    simon:VecGeom.build blyth$ 






EOU
}
vecgeom-sdir(){ echo $(local-base)/env/geometry/VecGeom ; }
vecgeom-bdir(){ echo $(vecgeom-dir).build ; }
vecgeom-idir(){ echo $(vecgeom-dir).install ; }
vecgeom-dir(){  echo $(vecgeom-sdir) ; }

vecgeom-cd(){   cd $(vecgeom-dir); }
vecgeom-scd(){  cd $(vecgeom-sdir); }
vecgeom-bcd(){  cd $(vecgeom-bdir); }
vecgeom-icd(){  cd $(vecgeom-idir); }

vecgeom-url(){ echo https://gitlab.cern.ch/VecGeom/VecGeom.git ; }
vecgeom-name(){ echo VecGeom ; }
vecgeom-get(){
   local dir=$(dirname $(vecgeom-dir)) &&  mkdir -p $dir && cd $dir
   [ ! -d "$(vecgeom-name)" ] && git clone $(vecgeom-url) 
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


