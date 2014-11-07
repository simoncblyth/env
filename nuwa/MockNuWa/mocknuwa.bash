# === func-gen- : nuwa/MockNuWa/mocknuwa fgp nuwa/MockNuWa/mocknuwa.bash fgn mocknuwa fgh nuwa/MockNuWa

#set -e

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
mocknuwa-cd(){  cd $(mocknuwa-sdir); }
mocknuwa-scd(){  cd $(mocknuwa-sdir); }
mocknuwa-tcd(){  cd $(mocknuwa-tdir); }


mocknuwa-cmake(){
   local iwd=$PWD
   mkdir -p $(mocknuwa-tdir)
   mocknuwa-tcd
   cmake $(mocknuwa-sdir) -DCMAKE_INSTALL_PREFIX=$(mocknuwa-prefix) -DCMAKE_BUILD_TYPE=Debug 
   cd $iwd
}
mocknuwa-make(){
   local iwd=$PWD
   mocknuwa-tcd
   make $*
   local rc=$?
   cd $iwd
   [ $rc -ne 0 ] && echo $FUNCNAME failed && return $rc
   return 0
}
mocknuwa-install(){
   mocknuwa-make install
}
mocknuwa-build(){
   mocknuwa-cmake
   #mocknuwa-make
   mocknuwa-install
}
mocknuwa-wipe(){
   rm -rf $(mocknuwa-tdir)
}
mocknuwa-build-full(){
   local iwd=$PWD
   mocknuwa-wipe
   mocknuwa-build
   cd $iwd
}

mocknuwa-runenv(){
   csa-
   csa-export

   export-
   export-export   # needed for template envvar for CPL saving 

   #export G4DAECHROMA_CLIENT_CONFIG=tcp://localhost:5001    # client to local broker
   export G4DAECHROMA_CLIENT_CONFIG=""
}

mocknuwa--(){
   mocknuwa-make install
   [ $? -ne 0 ] && echo $FUNCNAME failed && return 1
   mocknuwa-run
}

mocknuwa-lldb(){
   lldb $(which MockNuWa)
}


mocknuwa-run(){
   local bin=$(which MockNuWa)
   mocknuwa-runenv 
   env | grep G4DAECHROMA
   ls -l $bin
   $bin
}





