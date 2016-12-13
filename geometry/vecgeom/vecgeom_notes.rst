VecGeom Notes
================

Thoughts on VecGeom GPU from a very brief usage attempt and code look
------------------------------------------------------------------------

* too stuck in Geant4 geometry model mindset 

  * tiny volumes are one-by-one copied to GPU 
  * stuck in the each shape is an object mindset,
    on the GPU or just when you have many thousands of volumes it 
    makes more sense to think lower level : entries in a buffer 
  * looks like no repeated volume instancing in the
    graphics sense 

* contrasting the way OptiX geometry is setup to VecGeom it 
  seems plain that VecGeom aint going to be interfaceable 
  into the OptiX/Opticks way of geometry handling

* VecGeom implemention may however help with ideas
  on how to structure a less manual analytic geometry 
  approach

* examining VecGeom makes me appreciate what OptiX provides

* also the very particular way OptiX needs intersection code setup::


OptiX intersection
--------------------

Very particular tmin/tmax squeeze approach implemented via rtPotentialIntersection, rtReportIntersection pairs.
Other than intersection algorithm ideas there seems little way that VecGeom could help.::

     884 static __device__
     885 void intersect_box(quad& q0, quad& q1, quad& q2, quad& q3, const uint4& identity)
     886 {
     887 
     888   const float3 min_ = make_float3(q0.f.x - q0.f.w, q0.f.y - q0.f.w, q0.f.z - q0.f.w );
     889   const float3 max_ = make_float3(q0.f.x + q0.f.w, q0.f.y + q0.f.w, q0.f.z + q0.f.w );
     890   const float3 cen_ = make_float3(q0.f.x, q0.f.y, q0.f.z) ;
     891 
     892   float3 t0 = (min_ - ray.origin)/ray.direction;
     893   float3 t1 = (max_ - ray.origin)/ray.direction;
     894 
     895   // slab method 
     896   float3 near = fminf(t0, t1);
     897   float3 far = fmaxf(t0, t1);
     898   float tmin = fmaxf( near );
     899   float tmax = fminf( far );
     900 
     901   if(tmin <= tmax && tmax > 0.f)
     902   {
     903       bool check_second = true;
     904       float tint = tmin > 0.f ? tmin : tmax ;
     905 
     906       if(rtPotentialIntersection(tint))
     907       {
     908           float3 p = ray.origin + tint*ray.direction - cen_ ;
     909           float3 pa = make_float3(fabs(p.x), fabs(p.y), fabs(p.z)) ;
     910           float pmax = fmaxf(pa);
     911 
     912           float3 n = make_float3(0.f) ;
     913           if(      pa.x >= pa.y && pa.x >= pa.z ) n.x = copysignf( 1.f , p.x ) ;
     914           else if( pa.y >= pa.x && pa.y >= pa.z ) n.y = copysignf( 1.f , p.y ) ;
     915           else if( pa.z >= pa.x && pa.z >= pa.y ) n.z = copysignf( 1.f , p.z ) ;
     916 
     917 
     918           shading_normal = geometric_normal = n ;
     919           instanceIdentity = identity ;
     920           if(rtReportIntersection(0)) check_second = false ;   // material index 0 
     921       }
     922 
     923       // handle when inside box, or are epsilon near clipped 
     924       if(check_second)
     925       {
     926           if(rtPotentialIntersection(tmax))





Refs
-----

* http://indico.cern.ch/event/289682/contributions/664274/attachments/540916/745663/johannes_concurrency_forum_3.pdf


Cuda Usage in VecGeom
-----------------------

::

    simon:VecGeom blyth$ find . -type f -exec grep -H CudaManager.h {} \;
    ./management/CudaManager.h:/// \file CudaManager.h
    ./management/CudaManager_0.h:/// \file CudaManager.h
    ./navigation/NavigationState.h:#include "management/CudaManager.h"
    ./navigation/NavStatePool.h:#include "management/CudaManager.h"
    ./source/benchmarking/Benchmarker.cpp:#include "management/CudaManager.h"
    ./source/benchmarking/Benchmarker.cu:#include "management/CudaManager.h"
    ./source/benchmarking/NavigationBenchmarker.cpp:#include "management/CudaManager.h"
    ./source/benchmarking/NavigationBenchmarker.cu:#include "management/CudaManager.h"
    ./source/CudaManager.cpp:#include "management/CudaManager.h"
    ./source/CudaManager.cu:#include "management/CudaManager.h"
    ./source/UnplacedBooleanVolume.cpp:#include "management/CudaManager.h"
    ./source/UnplacedScaledShape.cpp:#include "management/CudaManager.h"
    ./test/benchmark/OrbBenchmark.cpp:#include "management/CudaManager.h"
    ./test/core/create_geometry.cpp:#include "management/CudaManager.h"
    ./test/core/TestNavigationStatePool.cpp:#include "management/CudaManager.h"
    ./test/root/ImportFromRootFileTest.cpp:#include "management/CudaManager.h"
    ./userexamples/src/TestNavigationStatePool.cu:#include "management/CudaManager.h"


Benchmark asserts 
-------------------

::

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

::

    simon:VecGeom blyth$ find . -type f -exec grep -H PlacedVolume\ created\ after\ geometry\ is\ closed {} \;
    ./source/GeoManager.cpp:    std::cerr << "PlacedVolume created after geometry is closed --> will not be registered\n";




::

    simon:VecGeom blyth$ find . -type f -exec grep -H gCompactPlacedVolBuffer {} \;
    ./management/CudaManager.h:// extern __device__ VPlacedVolume *gCompactPlacedVolBuffer;
    ./management/GeoManager.h:  static VPlacedVolume *gCompactPlacedVolBuffer;
    ./navigation/NavigationState.h:    return &vecgeom::GeoManager::gCompactPlacedVolBuffer[index];
    ./navigation/NavigationState.h:    // (failed preveously due to undefined symbol vecgeom::cuda::GeoManager::gCompactPlacedVolBuffer)
    ./services/NavigationSpecializer.cpp:    outstream << "VPlacedVolume const * pvol = &GeoManager::gCompactPlacedVolBuffer[" << fTargetVolIds[transitionid]

    ./source/CudaGlobalSymbols.cu:__device__ VPlacedVolume *gCompactPlacedVolBuffer;

    ./source/CudaManager.cpp:           (size_t)(&GeoManager::gCompactPlacedVolBuffer[0]) + sizeof(vecgeom::cxx::VPlacedVolume) * (*i)->id());
    ./source/CudaManager.cpp:    VPlacedVolume* vpv = &GeoManager::gCompactPlacedVolBuffer[i] ; 
    ./source/CudaManager.cpp:    //assert(&GeoManager::gCompactPlacedVolBuffer[i] != nullptr);
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




Contrast VecGeom GPU geo setup with oxrap/OScene.cc
--------------------------------------------------------

oxrap/OScene::init

* sets up OptiX context with geometry info, including GPU textures


Observations of VecGeom. Hmm no logging, no profiling machinery, copying volumes to GPU 1-by-1, no instancing, not good signs.::

    053 vecgeom::DevicePtr<const vecgeom::cuda::VPlacedVolume> CudaManager::Synchronize()
     54 {
     55   Stopwatch timer, overalltimer;
     56   overalltimer.Start();
     57   if (verbose_ > 0) std::cerr << "Starting synchronization to GPU.\n";
     58 
     59   // Will return null if no geometry is loaded
     60   if (synchronized) return vecgeom::DevicePtr<const vecgeom::cuda::VPlacedVolume>(world_gpu_);
     61 
     62   CleanGpu();
     63 
     64   // Populate the memory map with GPU addresses
     65 
     66   AllocateGeometry();
     67 
     68   // Create new objects with pointers adjusted to point to GPU memory, then
     69   // copy them to the allocated memory locations on the GPU.
     70 
     71   if (verbose_ > 1) std::cerr << "Copying geometry to GPU..." << std::endl;
     72 
     73   if (verbose_ > 2) std::cerr << "\nCopying logical volumes...";
     74   timer.Start();
     75   for (std::set<LogicalVolume const *>::const_iterator i = logical_volumes_.begin(); i != logical_volumes_.end(); ++i) {
     76 
     77     (*i)->CopyToGpu(LookupUnplaced((*i)->GetUnplacedVolume()), LookupDaughters((*i)->fDaughters), LookupLogical(*i));
     78   }
     79   timer.Stop();
     80   if (verbose_ > 2) std::cerr << " OK; TIME NEEDED " << timer.Elapsed() << "s \n";
     81 
     82   if (verbose_ > 2) std::cerr << "Copying unplaced volumes...";
     83   timer.Start();
     84   for (std::set<VUnplacedVolume const *>::const_iterator i = unplaced_volumes_.begin(); i != unplaced_volumes_.end();
     85        ++i) {
     86 
     87     (*i)->CopyToGpu(LookupUnplaced(*i));
     88   }
     89   timer.Stop();
     90   if (verbose_ > 2) std::cout << " OK; TIME NEEDED " << timer.Elapsed() << "s \n";
     91 
     92   if (verbose_ > 2) std::cout << "Copying transformations_...";
     93   timer.Start();
     94   for (std::set<Transformation3D const *>::const_iterator i = transformations_.begin(); i != transformations_.end();
     95        ++i) {
     96 
     97     (*i)->CopyToGpu(LookupTransformation(*i));
     98   }
     99   timer.Stop();
    100   if (verbose_ > 2) std::cout << " OK; TIME NEEDED " << timer.Elapsed() << "s \n";
    101 
    102   if (verbose_ > 2) std::cout << "Copying placed volumes...";
    103   // TODO: eventually we want to copy the placed volumes in one go (since they live now in contiguous buffers on both
    104   // sides
    105   // (the catch is that we will need to fix the virtual table pointers on the device side manually )
    106 
    107   timer.Start();
    108   for (std::set<VPlacedVolume const *>::const_iterator i = placed_volumes_.begin(); i != placed_volumes_.end(); ++i) {
    109 
    110     (*i)->CopyToGpu(LookupLogical((*i)->GetLogicalVolume()), LookupTransformation((*i)->GetTransformation()),
    111                     LookupPlaced(*i));
    112 




