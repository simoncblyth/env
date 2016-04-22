# === func-gen- : OPTICKS fgp ./OPTICKS.bash fgn OPTICKS fgh .
OPTICKS-(){         source $(OPTICKS-source) ; }
OPTICKS-src(){      echo OPTICKS.bash ; }
OPTICKS-source(){   echo ${BASH_SOURCE:-$(env-home)/$(OPTICKS-src)} ; }
OPTICKS-vi(){       vi $(OPTICKS-source) ; }
OPTICKS-env(){      elocal- ; }
OPTICKS-usage(){ cat << EOU

OPTICKS : experiment with umbrella cmake building
====================================================

Aiming for this to go in top level of a new opticks repo
together with top level CMakeLists.txt

Intend to allow building independent of the env.


Dependencies
--------------

::

   =====================  ===============  =============   ==============================================================================
   directory              precursor        pkg name        required find package 
   =====================  ===============  =============   ==============================================================================
   boost/bpo/bcfg         bcfg-            Cfg             Boost
   boost/bregex           bregex-          Bregex          Boost
   graphics/ppm           ppm-             PPM             
   numerics/npy           npy-             NPY             Boost GLM Bregex 
   opticks                opticks-         Opticks         Boost GLM Bregex Cfg NPY 
   optix/ggeo             ggeo-            GGeo            Boost GLM Bregex Cfg NPY Opticks
   graphics/assimpwrap    assimpwrap-      AssimpWrap      Boost Assimp GGeo GLM NPY Opticks
   graphics/openmeshrap   openmeshrap-     OpenMeshRap     Boost GLM NPY GGeo Opticks OpenMesh 
   graphics/oglrap        oglrap-          OGLRap          GLEW GLFW GLM Boost Cfg Opticks GGeo PPM NPY Bregex ImGui        
   cuda/cudawrap          cudawrap-        CUDAWrap        CUDA (ssl)
   numerics/thrustrap     thrustrap-       ThrustRap       CUDA Boost GLM NPY CUDAWrap 
   graphics/optixrap      optixrap-        OptiXRap        OptiX CUDA Boost GLM NPY Opticks Assimp AssimpWrap GGeo CUDAWrap ThrustRap 
   opticksop              opop-            OpticksOp       OptiX CUDA Boost GLM Cfg Opticks GGeo NPY OptiXRap CUDAWrap ThrustRap      
   opticksgl              opgl-            OpticksGL       OptiX CUDA Boost GLM GLEW GLFW OGLRap NPY Opticks Assimp AssimpWrap GGeo CUDAWrap ThrustRap OptiXRap OpticksOp
   graphics/ggeoview      ggv-             GGeoView        OptiX CUDA Boost GLM GLEW GLFW OGLRap NPY Cfg Opticks 
                                                           Assimp AssimpWrap OpenMesh OpenMeshRap GGeo ImGui Bregex OptiXRap CUDAWrap ThrustRap OpticksOp OpticksGL 
   optix/cfg4             cfg4-            CfG4            Boost Bregex GLM NPY Cfg GGeo Opticks Geant4 EnvXercesC G4DAE 
   =====================  ===============  =============   ==============================================================================


Collective needs installs
--------------------------

::

    simon:env blyth$ OPTICKS-make
    [  1%] Built target Cfg
    [  2%] Built target Bregex
    [  4%] Built target regexsearchTest
    [  4%] Building CXX object numerics/npy/CMakeFiles/NPY.dir/NPYBase.cpp.o
    /Users/blyth/env/numerics/npy/NPYBase.cpp:13:10: fatal error: 'regexsearch.hh' file not found
    #include "regexsearch.hh"
             ^
    1 error generated.
    make[2]: *** [numerics/npy/CMakeFiles/NPY.dir/NPYBase.cpp.o] Error 1
    make[1]: *** [numerics/npy/CMakeFiles/NPY.dir/all] Error 2
    make: *** [all] Error 2
    simon:env blyth$ 


NPY was needing Bregex installed headers, so it fails on first run ?

* Workarounds look complicated, see cmake-
* Pragmatically adjust all internal _INCLUDE_DIRS to source directories rather than install directories. 
* That seemed to work for a while, but after a wipe OPTICKS-cmake fails as all the _LIBRARIES 
  come back NOTFOUND at config time


find_package assumes do not need to build/install it ? 
---------------------------------------------------------

Workaround this using SUPERBUILD variable, as explained in FindBregex.cmake::

    if(SUPERBUILD)
        if(NOT Bregex_LIBRARIES)
           set(Bregex_LIBRARIES Bregex)
        endif()
    endif(SUPERBUILD)

    # When no lib is found at configure time : ie when cmake is run
    # find_package normally yields NOTFOUND
    # but here if SUPERBUILD is defined Bregex_LIBRARIES
    # is set to the target name: Bregex. 
    # 
    # This allows the build to proceed if the target
    # is included amongst the add_subdirectory of the super build.
    #

Hmm getting cycle warnings
----------------------------

All due to interloper directory /usr/local/env/numerics/npy/lib::

    NOT WITH_NPYSERVER
    -- Configuring done
    CMake Warning at optix/ggeo/CMakeLists.txt:51 (add_library):
      Cannot generate a safe runtime search path for target GGeo because there is
      a cycle in the constraint graph:

        dir 0 is [/opt/local/lib]
        dir 1 is [/usr/local/env/numerics/npy/lib]
          dir 4 must precede it due to runtime library [libNPY.dylib]
        dir 2 is [/usr/local/opticks/build/ALL/opticks]
        dir 3 is [/usr/local/opticks/build/ALL/boost/bpo/bcfg]
        dir 4 is [/usr/local/opticks/build/ALL/numerics/npy]
          dir 1 must precede it due to runtime library [libNPY.dylib]
        dir 5 is [/usr/local/opticks/build/ALL/boost/bregex]

      Some of these libraries may not be found correctly.

::

    simon:env blyth$ otool-;otool-rpath /usr/local/opticks/bin/GGeoView  | grep path | uniq
         path /usr/local/cuda/lib (offset 12)
         path /usr/local/opticks/lib (offset 12)
         path /opt/local/lib (offset 12)
         path /usr/local/env/graphics/glew/1.12.0/lib (offset 12)
         path /usr/local/env/numerics/npy/lib (offset 12)
         path /usr/local/env/graphics/OpenMesh/4.1/lib (offset 12)
         path /usr/local/env/graphics/gui/imgui.install/lib (offset 12)
         path /Developer/OptiX/lib64 (offset 12)


This issue went away, pilot error ?


Install seems to build again ?
-----------------------------------

::

  614  rm -rf /usr/local/opticks/*
  615  OPTICKS-
  616  OPTICKS-cmake
  617  OPTICKS-make
  625  OPTICKS-install
  626  otool-
  627  otool-rpath /usr/local/opticks/bin/GGeoView 
  628  /usr/local/opticks/bin/GGeoView /tmp/g4_00.dae



Handling tests
----------------

All tests are bundled into /usr/local/opticks/bin/


Usage
-------

::

   . OPTICKS.bash 

   OPTICKS-cmake
   OPTICKS-install

   OPTICKS-run


Pristine cycle::

   e;. OPTICKS.bash;OPTICKS-wipe;OPTICKS-cmake;OPTICKS-install


See also cmake-


To Consider
-------------

* conditional cfg4 build depending on finding G4 

* externals depend on env bash functions for getting and installing
  and env cmake modules for finding 

* externals gathering


ggv.sh Launcher
-----------------

Bash launcher ggv.sh tied into the individual bash functions and 
sets up envvars::

   OPTICKS_GEOKEY
   OPTICKS_QUERY
   OPTICKS_CTRL
   OPTICKS_MESHFIX
   OPTICKS_MESHFIX_CFG

* OpticksResource looks for a metadata sidecar .ini accompanying the .dae
  eg for /tmp/g4_00.dae the file /tmp/g4_00.ini is looked for

* TODO: enable all envvars to come in via the metadata .ini approach with 
  potential to be overridded by the envvars 


Umbrella CMakeLists.txt
-------------------------

* avoid tests in different pkgs with same name 

Thoughts
--------

The umbrella cmake build avoids using the bash functions
for each of the packages... but those are kinda useful
for development. 


Interference between granular and collective builds
-----------------------------------------------------

The collective OPTICKS-cmake is setting CMAKE_INSTALL_PREFIX
to /usr/local/opticks which differs from
the one in the pkg bash functions ?

The collective build misses the opticks lib symbols
because linking to outdated /usr/local/opticks/lib/libOpticks.dylib
why didnt this get updated ?

Maybe need to arrange common install dir between the granular and collective ?

Checking the linking commandline ~/chk there is mixture between 
where the libs are coming from ? Why ?

Suspect name issue wrt NPY and npy 
and Opticks and opticks.

Directory name and project name need to match ? 

RPATH/library confusion
-------------------------

RPATH covers all dirs including the collective, 
unclear which libs are getting used::

    simon:env blyth$ otool-;otool-rpath /usr/local/opticks/bin/GGeoView | grep path | uniq
         path /usr/local/cuda/lib (offset 12)
         path /usr/local/opticks/lib (offset 12)
         path /opt/local/lib (offset 12)
         path /usr/local/env/graphics/glew/1.12.0/lib (offset 12)
         path /usr/local/env/boost/bpo/bcfg/lib (offset 12)
         path /usr/local/env/boost/bregex/lib (offset 12)
         path /usr/local/env/numerics/npy/lib (offset 12)
         path /usr/local/env/opticks/lib (offset 12)
         path /usr/local/env/graphics/assimpwrap/lib (offset 12)
         path /usr/local/env/graphics/OpenMesh/4.1/lib (offset 12)
         path /usr/local/env/graphics/openmeshrap/lib (offset 12)
         path /usr/local/env/optix/ggeo/lib (offset 12)
         path /usr/local/env/graphics/gui/imgui.install/lib (offset 12)
         path /usr/local/env/graphics/oglrap/lib (offset 12)
         path /usr/local/env/cuda/CUDAWrap/lib (offset 12)
         path /usr/local/env/graphics/OptiXRap/lib (offset 12)
         path /usr/local/env/numerics/ThrustRap/lib (offset 12)
         path /usr/local/env/opticksop/lib (offset 12)
         path /usr/local/env/opticksgl/lib (offset 12)
         path /Developer/OptiX/lib64 (offset 12)


* collective build installs everything to  -DCMAKE_INSTALL_PREFIX=$(local-base)/opticks 
* individual builds install many places eg  -DCMAKE_INSTALL_PREFIX=$(local-base)/env/boost/bpo/bcfg  etc...
* FindX.cmake returns the individual locations

Centralized approach ?
~~~~~~~~~~~~~~~~~~~~~~~~

* adopt centralized location for individual builds...

  * change all the FindX.cmake to give centralized position
  * nice and simple
  * BUT: means common namespace, so should improve classname prefixing  

* individual and collective builds that operate in the same pot 

* how to handle internal/external distinction ? which changes as pkg matured

EOU
}

OPTICKS-dirs(){  cat << EOL
boost/bpo/bcfg
boost/bregex
numerics/npy
opticks
optix/ggeo
graphics/assimpwrap
graphics/openmeshrap
graphics/oglrap
cuda/cudawrap
numerics/thrustrap
graphics/optixrap
opticksop
opticksgl
graphics/ggeoview
EOL
}

OPTICKS-internals(){  cat << EOI
Cfg
Bregex
PPM
NPY
Opticks
GGeo
AssimpWrap
OpenMeshRap
OGLRap
CUDAWrap
ThrustRap
OptiXRap
OpticksOp
OpticksGL
OptiXThrust
NumpyServer
EOI
}
OPTICKS-xternals(){  cat << EOX
Boost
GLM
EnvXercesC
G4DAE
Assimp
OpenMesh
GLEW
GLEQ
GLFW
ImGui
ZMQ
AsioZMQ
EOX
}
OPTICKS-other(){  cat << EOO
OpenVR
CNPY
NuWaCLHEP
NuWaGeant4
cJSON
RapSqlite
SQLite3
ChromaPhotonList
G4DAEChroma
NuWaDataModel
ChromaGeant4CLHEP
CLHEP
ROOT
ZMQRoot
EOO
}


OPTICKS-tfind-(){ 
  local f
  local base=$ENV_HOME/CMake/Modules
  OPTICKS-${1} | while read f 
  do
     echo $base/Find${f}.cmake
  done 
}

OPTICKS-ifind(){ vi $(OPTICKS-tfind- internals) ; }
OPTICKS-xfind(){ vi $(OPTICKS-tfind- xternals) ; }
OPTICKS-ofind(){ vi $(OPTICKS-tfind- other) ; }


OPTICKS-dir(){ echo $(local-base)/opticks ; }
OPTICKS-cd(){  cd $(OPTICKS-dir); }


OPTICKS-sdir(){ echo $(env-home) ; }
OPTICKS-idir(){ echo $(local-base)/opticks ; }
OPTICKS-bdir(){ echo $(local-base)/opticks/build/${1:-ALL} ; }

OPTICKS-scd(){  cd $(OPTICKS-sdir); }
OPTICKS-cd(){   cd $(OPTICKS-sdir); }
OPTICKS-icd(){  cd $(OPTICKS-idir); }
OPTICKS-bcd(){  cd $(OPTICKS-bdir); }



OPTICKS-edit(){ cd $ENV_HOME ; vi $(OPTICKS-cmakelists) ; }
OPTICKS-cmakelists(){
  local dir
  OPTICKS-dirs | while read dir 
  do
      echo $dir/CMakeLists.txt
  done
}

OPTICKS-wipe(){
   local bdir=$(OPTICKS-bdir)
   rm -rf $bdir
}






OPTICKS-optix-install-dir(){ echo /Developer/OptiX ; }
#OPTICKS-optix-install-dir(){ echo -n ; }


OPTICKS-cmake(){
   local msg="=== $FUNCNAME : "
   local iwd=$PWD

   local bdir=$(OPTICKS-bdir)
   mkdir -p $bdir

   [ ! -d "$bdir" ] && echo $msg NO bdir $bdir && return  

   OPTICKS-bcd
   cmake \
       -DWITH_OPTIX:BOOL=ON \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(OPTICKS-idir) \
       -DOptiX_INSTALL_DIR=$(OPTICKS-optix-install-dir) \
       $(OPTICKS-sdir)

   cd $iwd
}

OPTICKS-bin(){ echo $(OPTICKS-idir)/bin/GGeoView ; }


OPTICKS-make(){
   local iwd=$PWD

   OPTICKS-bcd
   make $*

   cd $iwd
}

OPTICKS-install(){
   OPTICKS-make install
}


OPTICKS--()
{
    OPTICKS-wipe
    OPTICKS-cmake
    OPTICKS-make
    OPTICKS-install
}

OPTICKS-run()
{

    export-
    export-export   ## needed to setup DAE_NAME_DYB the envvar name pointed at by the default OPTICKS_GEOKEY 

    local bin=$(OPTICKS-bin)
    $bin $*         ## bare running with no bash script, for checking defaults 

}
