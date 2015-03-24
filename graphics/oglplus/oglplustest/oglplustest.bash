oglplustest-src(){      echo graphics/oglplus/oglplustest/oglplustest.bash ; }
oglplustest-source(){   echo ${BASH_SOURCE:-$(env-home)/$(oglplustest-src)} ; }
oglplustest-vi(){       vi $(oglplustest-source) ; }
oglplustest-env(){      elocal- ; }
oglplustest-usage(){ cat << EOU


cp /usr/local/env/graphics/oglplus/oglplus-0.59.0/example/standalone/029_mandelbrot_glfw3.cpp .


EOU
}


oglplustest-sdir(){ echo $(env-home)/graphics/oglplus/oglplustest ; }
oglplustest-idir(){ echo $(local-base)/env/graphics/oglplus/oglplustest ; }
oglplustest-bdir(){ echo $(oglplustest-idir).build ; }

oglplustest-scd(){  cd $(oglplustest-sdir); }
oglplustest-cd(){  cd $(oglplustest-sdir); }

oglplustest-icd(){  cd $(oglplustest-idir); }
oglplustest-bcd(){  cd $(oglplustest-bdir); }
oglplustest-name(){ echo OGLPlusTest ; }

oglplustest-wipe(){
   local bdir=$(oglplustest-bdir)
   rm -rf $bdir
}

oglplustest-cmake(){
   local iwd=$PWD

   local bdir=$(oglplustest-bdir)
   mkdir -p $bdir
  
   oglplustest-bcd 
   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(oglplustest-idir) \
       $(oglplustest-sdir)

   cd $iwd
}

oglplustest-make(){
   local iwd=$PWD

   oglplustest-bcd 
   make $*

   cd $iwd
}

oglplustest-install(){
   oglplustest-make install
}

oglplustest-bin(){ echo $(oglplustest-idir)/bin/$(oglplustest-name) ; }


