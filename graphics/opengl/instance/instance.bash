# === func-gen- : graphics/opengl/instance/instance fgp graphics/opengl/instance/instance.bash fgn instance fgh graphics/opengl/instance
instance-src(){      echo graphics/opengl/instance/instance.bash ; }
instance-source(){   echo ${BASH_SOURCE:-$(env-home)/$(instance-src)} ; }
instance-vi(){       vi $(instance-source) ; }
instance-usage(){ cat << EOU
OpenGL Instancing 
====================

EOU
}

instance-env(){      elocal- ; } 

instance-sdir(){ echo $(env-home)/graphics/opengl/instance ; }
instance-idir(){ echo $(local-base)/env/graphics/opengl/instance ; }
instance-bdir(){ echo $(instance-idir).build ; }
instance-bindir(){ echo $(instance-idir)/bin ; }

instance-cd(){   cd $(instance-sdir); }
instance-c(){    cd $(instance-sdir); }
instance-icd(){  cd $(instance-idir); }
instance-bcd(){  cd $(instance-bdir); }

instance-name(){ echo INSTANCE ; }

instance-wipe(){
   local bdir=$(instance-bdir)
   rm -rf $bdir
}

instance-cmake(){
   local iwd=$PWD

   local bdir=$(instance-bdir)
   mkdir -p $bdir
 
   opticks- 
 
   instance-bcd 
   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
       $(instance-sdir)

   cd $iwd
}

instance-make(){
   local iwd=$PWD

   instance-bcd
   make $*
   cd $iwd
}

instance-install(){
   instance-make install
}

instance--()
{
    instance-wipe
    instance-cmake
    instance-make
    instance-install
}


