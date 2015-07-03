# === func-gen- : graphics/thrust_opengl_interop/throgl fgp graphics/thrust_opengl_interop/throgl.bash fgn throgl fgh graphics/thrust_opengl_interop
throgl-src(){      echo graphics/thrust_opengl_interop/throgl.bash ; }
throgl-source(){   echo ${BASH_SOURCE:-$(env-home)/$(throgl-src)} ; }
throgl-vi(){       vi $(throgl-source) ; }
throgl-usage(){ cat << EOU




EOU
}

throgl-env(){      elocal- ; }

throgl-sdir(){ echo $(env-home)/graphics/thrust_opengl_interop ; }
throgl-idir(){ echo $(local-base)/env/graphics/thrust_opengl_interop ; }
throgl-bdir(){ echo $(throgl-idir).build ; }
throgl-bindir(){ echo $(throgl-idir)/bin ; }

throgl-scd(){  cd $(throgl-sdir); }
throgl-cd(){   cd $(throgl-sdir); }

throgl-icd(){  cd $(throgl-idir); }
throgl-bcd(){  cd $(throgl-bdir); }
throgl-name(){ echo ThrustOpenGLInterop ; }

throgl-wipe(){
   local bdir=$(throgl-bdir)
   rm -rf $bdir
}

throgl-cmake(){
   local iwd=$PWD

   local bdir=$(throgl-bdir)
   mkdir -p $bdir
  
   throgl-bcd 
   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(throgl-idir) \
       $(throgl-sdir)

   cd $iwd
}


throgl-make(){
   local iwd=$PWD

   throgl-bcd
   make $*
   cd $iwd
}

throgl-install(){
   throgl-make install
}

throgl--()
{
    throgl-wipe
    throgl-cmake
    throgl-make
    throgl-install

}

