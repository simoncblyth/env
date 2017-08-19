instcull-src(){      echo graphics/opengl/instcull/instcull.bash ; }
instcull-source(){   echo ${BASH_SOURCE:-$(env-home)/$(instcull-src)} ; }
instcull-vi(){       vi $(instcull-source) ; }
instcull-usage(){ cat << EOU
OpenGL Instance Culling
========================

EOU
}

instcull-env(){      elocal- ; } 

instcull-sdir(){ echo $(env-home)/graphics/opengl/instcull ; }
instcull-idir(){ echo $(local-base)/env/graphics/opengl/instcull ; }
instcull-bdir(){ echo $(instcull-idir).build ; }
instcull-bindir(){ echo $(instcull-idir)/bin ; }

instcull-cd(){   cd $(instcull-sdir); }
instcull-c(){    cd $(instcull-sdir); }
instcull-icd(){  cd $(instcull-idir); }
instcull-bcd(){  cd $(instcull-bdir); }

instcull-name(){ echo INSTCULL ; }

instcull-wipe(){
   local bdir=$(instcull-bdir)
   rm -rf $bdir
}

instcull-cmake(){
   local iwd=$PWD

   local bdir=$(instcull-bdir)
   mkdir -p $bdir
 
   opticks- 
 
   instcull-bcd 
   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
       $(instcull-sdir)

   cd $iwd
}

instcull-make(){
   local iwd=$PWD

   instcull-bcd
   make $*
   cd $iwd
}

instcull-install(){
   instcull-make install
}

instcull--()
{
    instcull-wipe
    instcull-cmake
    instcull-make
    instcull-install
}


