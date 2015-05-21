# === func-gen- : graphics/gui/imguitest/imguitest fgp graphics/gui/imguitest/imguitest.bash fgn imguitest fgh graphics/gui/imguitest
imguitest-src(){      echo graphics/gui/imguitest/imguitest.bash ; }
imguitest-source(){   echo ${BASH_SOURCE:-$(env-home)/$(imguitest-src)} ; }
imguitest-vi(){       vi $(imguitest-source) ; }
imguitest-env(){      elocal- ; }
imguitest-usage(){ cat << EOU



EOU
}

imguitest-idir(){ echo $(local-base)/env/graphics/gui/imguitest.install ; }
imguitest-bdir(){ echo $(local-base)/env/graphics/gui/imguitest.build   ; }
imguitest-sdir(){ echo $(env-home)/graphics/gui/imguitest ; }

imguitest-icd(){  cd $(imguitest-idir); }
imguitest-bcd(){  cd $(imguitest-bdir); }
imguitest-scd(){  cd $(imguitest-sdir); }

imguitest-cd(){  cd $(imguitest-sdir); }


imguitest-wipe(){
   local bdir=$(imguitest-bdir)
   rm -rf $bdir
}

imguitest-cmake(){
   local bdir=$(imguitest-bdir)
   mkdir -p $bdir
   imguitest-bcd
   cmake $(imguitest-sdir) -DCMAKE_INSTALL_PREFIX=$(imguitest-idir) -DCMAKE_BUILD_TYPE=Debug 
}

imguitest-make(){
    local iwd=$PWD
    imguitest-bcd
    make $*
    cd $iwd
}

imguitest-install(){
   imguitest-make install
}

imguitest--(){
   imguitest-wipe
   imguitest-cmake
   imguitest-make
   [ $? -ne 0 ] && echo $FUNCNAME ERROR && return 1
   imguitest-install $*
}




