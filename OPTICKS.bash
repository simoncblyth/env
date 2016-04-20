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


* pull out classes from oglrap- that do not depend
  on OpenGL and place into opticks

* conditional build depending on finding OptiX

  * package or preprocessor partialized pkg to provide 
    viewing functionality without OptiX 

* conditional cfg4 build depending on finding G4 

* externals depend on env bash functions for getting and installing
  and env cmake modules for finding 

* externals gathering, take a look at dybinst 

* bash launcher ggv.sh is tied into the individual bash functions

* enable no envvar operation, eg for running with small test geometries
  which dont have much need for the geocache 



Defaults for running with minimal envvars
-------------------------------------------

::

    export-export   ## needed to setup geokeys to find geometry files

    # default OPTICKS_GEOKEY is DAE_NAME_DYB so that envvar must point at the geometry file

    simon:env blyth$ /usr/local/opticks/bin/GGeoView



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


Dependencies
--------------

::


   =====================  ===============  =============   ==============================================================================
   directory              precursor        pkg name        required find package 
   =====================  ===============  =============   ==============================================================================
   boost/bpo/bcfg         bcfg-            Cfg             Boost
   boost/bregex           bregex-          bregex          Boost
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
    

See cmake-



EOU
}
OPTICKS-dir(){ echo $(local-base)/opticks ; }
OPTICKS-cd(){  cd $(OPTICKS-dir); }


OPTICKS-sdir(){ echo $(env-home) ; }
OPTICKS-idir(){ echo $(local-base)/opticks ; }
OPTICKS-bdir(){ echo $(local-base)/opticks/build ; }

OPTICKS-scd(){  cd $(OPTICKS-sdir); }
OPTICKS-cd(){   cd $(OPTICKS-sdir); }
OPTICKS-icd(){  cd $(OPTICKS-idir); }
OPTICKS-bcd(){  cd $(OPTICKS-bdir); }

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
