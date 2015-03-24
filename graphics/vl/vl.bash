# === func-gen- : graphics/vl/vl fgp graphics/vl/vl.bash fgn vl fgh graphics/vl
vl-src(){      echo graphics/vl/vl.bash ; }
vl-source(){   echo ${BASH_SOURCE:-$(env-home)/$(vl-src)} ; }
vl-vi(){       vi $(vl-source) ; }
vl-env(){      elocal- ; }
vl-usage(){ cat << EOU

Visualization Library
=======================

A lightweight C++ OpenGL middleware for 2D/3D graphics

* http://www.visualizationlibrary.org/documentation/pag_key_features.html

* http://www.visualizationlibrary.org/documentation/pag_guides.html

* https://github.com/Wulfire/visualizationlibrary


Looking for a C++ equivalent of pyopengl/glumpy to 
take the pain out of using OpenGL without compromising 
flexibility.



EOU
}
vl-sdir(){ echo $(local-base)/env/graphics/vl.src ; }
vl-bdir(){ echo $(local-base)/env/graphics/vl.build ; }
vl-idir(){ echo $(local-base)/env/graphics/vl ; }

vl-scd(){  cd $(vl-sdir); }
vl-cd(){   cd $(vl-sdir); }

vl-bcd(){  cd $(vl-bdir); }
vl-icd(){  cd $(vl-idir); }


vl-get(){
   local sdir=$(dirname $(vl-sdir)) &&  mkdir -p $sdir && cd $sdir
   local nam=$(basename $(vl-sdir))

   #[ ! -d $nam ] && svn co http://visualizationlibrary.googlecode.com/svn/trunk $nam

   [ ! -d $nam ] && git clone https://github.com/Wulfire/visualizationlibrary.git $nam

}


vl-cmake(){
   local iwd=$PWD

   local bdir=$(vl-bdir)
   mkdir -p $bdir 
    
   vl-bcd
   cmake -DCMAKE_INSTALL_PREFIX=$(vl-idir) $(vl-sdir)

   cd $iwd
}


vl-make(){
   local iwd=$PWD
   vl-bcd
   make $*

   cd $iwd
}

vl-install(){
   vl-make install
}

