# === func-gen- : OPTICKS fgp ./OPTICKS.bash fgn OPTICKS fgh .
OPTICKS-(){         source $(OPTICKS-source) ; }
OPTICKS-src(){      echo OPTICKS.bash ; }
OPTICKS-source(){   echo ${BASH_SOURCE:-$(env-home)/$(OPTICKS-src)} ; }
OPTICKS-vi(){       vi $(OPTICKS-source) ; }
OPTICKS-env(){      elocal- ; }
OPTICKS-usage(){ cat << EOU

OPTICKS : experiment with umbrella cmake building
====================================================

Aiming for this to go in top level of a new Opticks repo
together with top level superbuild CMakeLists.txt

Intend to allow building independent of the env.

See Also
----------

cmake-
    background on cmake

cmakex-
    documenting the development of the OPTICKS- cmake machinery 


TODO
-----

* rename Opticks subpackage to OpticksCore/OpticksBase/OpticksMd for model, to avoid shouting OPTICKS for top level pkg 
* tidy up ssl and crypto : maybe in NPY_LIBRARIES 
* CUDAWrap: adopt standard tests approach 
* standardize package names, Wrap to Rap for AssimpWrap and CUDAWrap
* externalize or somehow exclude from standard building the Rap pkgs, as fairly stable
* machinery for getting externals
* spawn opticks repository 

* investigate cpack


Dependencies of internals
---------------------------

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


Usage
-------

::

   . OPTICKS.bash 

   OPTICKS-cmake
   OPTICKS-install

   OPTICKS-run


Pristine cycle::

   e;. OPTICKS.bash;OPTICKS-wipe;OPTICKS-cmake;OPTICKS-install


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
OPTICKS-bdir(){ echo $(local-base)/opticks/build ; }
OPTICKS-tdir(){ echo /tmp/opticks ; }

OPTICKS-scd(){  cd $(OPTICKS-sdir); }
OPTICKS-cd(){   cd $(OPTICKS-sdir); }
OPTICKS-icd(){  cd $(OPTICKS-idir); }
OPTICKS-bcd(){  cd $(OPTICKS-bdir); }


OPTICKS-txt(){   cd $ENV_HOME ; vi $(OPTICKS-txt-list) ; }
OPTICKS-bash(){  cd $ENV_HOME ; vi $(OPTICKS-bash-list) ; }
OPTICKS-edit(){  cd $ENV_HOME ; vi $(OPTICKS-bash-list) $(OPTICKS-txt-list) ; } 

OPTICKS-txt-list(){
  local dir
  OPTICKS-dirs | while read dir 
  do
      echo $dir/CMakeLists.txt
  done
}

OPTICKS-bash-list(){
  local dir
  OPTICKS-dirs | while read dir 
  do
      local rel=$dir/$(basename $dir).bash
      if [ -f "$rel" ]; 
      then
          echo $rel
      else
          echo MISSING $rel
      fi
  done
}

OPTICKS-wipe(){
   local bdir=$(OPTICKS-bdir)
   rm -rf $bdir
}

OPTICKS-optix-install-dir(){ echo /Developer/OptiX ; }

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
