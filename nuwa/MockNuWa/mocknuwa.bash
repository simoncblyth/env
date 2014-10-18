# === func-gen- : nuwa/MockNuWa/mocknuwa fgp nuwa/MockNuWa/mocknuwa.bash fgn mocknuwa fgh nuwa/MockNuWa
mocknuwa-src(){      echo nuwa/MockNuWa/mocknuwa.bash ; }
mocknuwa-source(){   echo ${BASH_SOURCE:-$(env-home)/$(mocknuwa-src)} ; }
mocknuwa-vi(){       vi $(mocknuwa-source) ; }
mocknuwa-env(){      
   elocal- 

   rootsys-
   geant4sys-

   chroma-   # just the above not enough, TODO: work out what else is required 
}

mocknuwa-env-check(){
   env | grep GEANT4
   env | grep ROOT
   env | grep CLHEP 
}

mocknuwa-usage(){ cat << EOU

MockNuWa
=========




EOU
}


mocknuwa-prefix(){ echo $(local-base)/env/nuwa ; }
mocknuwa-dir(){ echo $(local-base)/env/nuwa/MockNuWa  ; }
mocknuwa-sdir(){ echo $(env-home)/nuwa/MockNuWa  ; }
mocknuwa-tdir(){ echo /tmp/nuwa/MockNuWa  ; }
mocknuwa-cd(){  cd $(mocknuwa-dir); }
mocknuwa-scd(){  cd $(mocknuwa-sdir); }
mocknuwa-tcd(){  cd $(mocknuwa-tdir); }


mocknuwa-cmake(){
   local iwd=$PWD
   mkdir -p $(mocknuwa-tdir)
   mocknuwa-tcd
   cmake $(mocknuwa-sdir) -DCMAKE_INSTALL_PREFIX=$(mocknuwa-prefix)
   cd $iwd
}
mocknuwa-make(){
   local iwd=$PWD
   mocknuwa-tcd
   make $*
   [ "$?" != "0" ] && echo $msg $FUNCNAME ERROR && return 1
   cd $iwd
}
mocknuwa-install(){
   mocknuwa-make install
}
mocknuwa-build(){
   mocknuwa-cmake
   mocknuwa-make
   mocknuwa-install
}
mocknuwa-wipe(){
   rm -rf $(mocknuwa-tdir)
}
mocknuwa-build-full(){
   mocknuwa-wipe
   mocknuwa-build
}

mocknuwa-build-and-run(){
   mocknuwa-make install
   [ $? -ne 0 ] && return 1 

   local bin=$(which MockNuWa)
   ls -l $bin
   $bin
}
mocknuwa--(){
   mocknuwa-  # update self
   mocknuwa-build-and-run
}


