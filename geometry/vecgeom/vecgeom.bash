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


Presentations
---------------


Sandro Wenzel : Towards a high performance geometry library for particle-detector simulation (ACAT 2014)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://indico.cern.ch/event/258092/contributions/1588561/attachments/454205/629615/ACAT2014GeometryTalkNewStyleV2.pdf
* ~/opticks_refs/ACAT2014GeometryTalkNewStyleV2.pdf



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


