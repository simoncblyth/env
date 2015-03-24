# === func-gen- : graphics/vl/vltest/vltest fgp graphics/vl/vltest/vltest.bash fgn vltest fgh graphics/vl/vltest
vltest-src(){      echo graphics/vl/vltest/vltest.bash ; }
vltest-source(){   echo ${BASH_SOURCE:-$(env-home)/$(vltest-src)} ; }
vltest-vi(){       vi $(vltest-source) ; }
vltest-usage(){ cat << EOU





EOU
}
vltest-env(){      
  elocal- 
  vl-
}

vltest-sdir(){ echo $(env-home)/graphics/vl/vltest ; }
vltest-idir(){ echo $(local-base)/env/graphics/vl/vltest ; }
vltest-bdir(){ echo $(vltest-idir).build ; }

vltest-scd(){  cd $(vltest-sdir); }
vltest-cd(){   cd $(vltest-sdir); }

vltest-icd(){  cd $(vltest-idir); }
vltest-bcd(){  cd $(vltest-bdir); }
vltest-name(){ echo VLTest ; }


vltest-wipe(){
   local bdir=$(vltest-bdir)
   rm -rf $bdir
}

vltest-cmake(){
   local iwd=$PWD

   local bdir=$(vltest-bdir)
   mkdir -p $bdir
  
   vltest-bcd 
   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DVL_ROOT=$(vl-idir)/cmake \
       -DCMAKE_INSTALL_PREFIX=$(vltest-idir) \
       $(vltest-sdir)

   cd $iwd
}

vltest-make(){
   local iwd=$PWD

   vltest-bcd
   make $*

   cd $iwd
}

vltest-install(){
   vltest-make install
}

vltest-bin(){ echo $(vltest-idir)/bin/$(vltest-name) ; }



vltest-get-app()
{
    local name=${1:-App_RotatingCube.hpp}
    local appd=$(vltest-sdir)/Applets
    mkdir -p $appd

    cp $(vl-sdir)/src/examples/Applets/$name $appd/$name
    cp $(vl-sdir)/src/examples/GLUT_example.cpp .



}


