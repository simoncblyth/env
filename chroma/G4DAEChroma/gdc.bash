# === func-gen- : chroma/G4DAEChroma/gdc fgp chroma/G4DAEChroma/gdc.bash fgn gdc fgh chroma/G4DAEChroma
gdc-src(){      echo chroma/G4DAEChroma/gdc.bash ; }
gdc-source(){   echo ${BASH_SOURCE:-$(env-home)/$(gdc-src)} ; }
gdc-vi(){       vi $(gdc-source) ; }
gdc-usage(){ cat << EOU

G4DAEChroma
============

EOU
}

gdc-name(){  echo G4DAEChroma ; }
gdc-dir(){   echo $(local-base)/env/chroma/$(gdc-name) ; }
gdc-prefix(){ echo $(gdc-dir) ; }

gdc-sdir(){ echo $(env-home)/chroma/$(gdc-name) ; }
gdc-bdir(){ echo /tmp/env/chroma/$(gdc-name) ; }

gdc-cd(){  cd $(gdc-sdir); }
gdc-icd(){  cd $(gdc-dir); }
gdc-scd(){  cd $(gdc-sdir); }
gdc-bcd(){  cd $(gdc-bdir); }


gdc-geant4-home(){
  case $NODE_TAG in
    D) echo /usr/local/env/chroma_env/src/geant4.9.5.p01 ;;
  esac
}
gdc-geant4-dir(){   
  case $NODE_TAG in
    D) echo /usr/local/env/chroma_env/lib/Geant4-9.5.1 ;;
  esac
}

gdc-env(){      
   elocal- ; 
   export GEANT4_HOME=$(gdc-geant4-home)

   rootsys-  # needed to find root headers
}

gdc-cmake(){
   type $FUNCNAME
   local iwd=$PWD
   mkdir -p $(gdc-bdir)
   gdc-bcd
   cmake -DGeant4_DIR=$(gdc-geant4-dir) \
         -DCMAKE_INSTALL_PREFIX=$(gdc-prefix) \
         -DCMAKE_BUILD_TYPE=Debug \
         $(gdc-sdir)

   cd $iwd
}
gdc-verbose(){ echo  1 ; }
gdc-make(){
   local iwd=$PWD
   gdc-bcd
   make $* VERBOSE=$(gdc-verbose)
   cd $iwd
}
gdc-install(){ gdc-make install ; }

gdc-build(){
   gdc-cmake
   gdc-make
   gdc-install
}
gdc-build-full(){
   gdc-wipe
   gdc-build
}
gdc-wipe(){
   rm -rf $(gdc-bdir)
}


############## running 

gdc-exe(){ echo $(gdc-prefix)/bin/$(gdc-name)Test ; }
gdc-run-env(){
  export-
  export-export
  #env | grep DAE
}
gdc-run(){
  $FUNCNAME-env
  local exe=$(gdc-exe)
  ls -l $exe 
  local cmd="$exe $*"
  echo $cmd
  eval $cmd
}


############## over to NuWa

gdc-nuwapkg(){    
  case $NODE_TAG in  
     N) echo $DYB/NuWa-trunk/dybgaudi/Utilities/$(gdc-name) ;;
     *) utilities- && echo $(utilities-dir)/$(gdc-name) ;;
  esac
}
  
gdc-nuwapkg-cd(){ cd $(gdc-nuwapkg)/$1 ; } 


gdc-names(){ 
   local path
   ls -1 $(gdc-sdir)/G4DAEChroma/*.hh  | while read path ; do
      local name=$(basename $path)
      echo ${name/.hh}
   done
}

gdc-nuwapkg-action-cmds(){
   local action=${1:-diff}
   local pkg=$(gdc-nuwapkg)
   local pkn=$(basename $pkg)
   local nam=$(gdc-name)

   cat << EOC
mkdir -p $pkg/$pkn
mkdir -p $pkg/src
EOC

   gdc-names |while read nam ; do
   cat << EOC
$action $(gdc-sdir)/$pkn/$nam.hh         $pkg/$pkn/$nam.hh
$action $(gdc-sdir)/src/$nam.cc          $pkg/src/$nam.cc
EOC
   done

}

gdc-nuwapkg-action(){
   local cmd
   $FUNCNAME-cmds $1 | while read cmd ; do
      echo $cmd
      eval $cmd
   done
}
gdc-nuwapkg-diff(){  gdc-nuwapkg-action diff ; }
gdc-nuwapkg-cpto(){  gdc-nuwapkg-action cp ; }



gdc-nuwacfg () 
{ 
    local msg="=== $FUNCNAME :";
    local pkg=$1;
    shift;
    [ ! -d "$pkg/cmt" ] && echo ERROR NO cmt SUBDIR && sleep 1000000;
    local iwd=$PWD;
    echo $msg for pkg $pkg;
    cd $pkg/cmt;
    cmt config;
    . setup.sh;
    cd $iwd
}



gdc-nuwaenv()
{   
    zmqroot-
    gdc-nuwacfg $(zmqroot-nuwapkg);
    cpl-
    gdc-nuwacfg $(cpl-nuwapkg);
    csa-
    gdc-nuwacfg $(csa-nuwapkg)
}

gdc-nuwapkg-make() 
{ 
    local iwd=$PWD;
    gdc-nuwaenv
    gdc-nuwapkg-cd cmt
    cmt config
    cmt make
    cd $iwd
}

