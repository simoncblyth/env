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

Need way to record the "split" event, as it slows down processing::

    2014-11-20 18:37:51,426 INFO    env.geant4.geometry.collada.g4daeview.daechromacontext:122 _set_rng_states
    2014-11-20 18:37:51,763 INFO    chroma.gpu.geometry :171 Splitting BVH between GPU and CPU memory at node 78800
    2014-11-20 18:37:51,786 INFO    chroma.gpu.geometry :201 device usage:




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


mocknuwa-db(){
   echo $LOCAL_BASE/env/nuwa/mocknuwa.db
}
mocknuwa-sqlite(){
   sqlite3 $(mocknuwa-db)
}

mocknuwa-runenv(){
   csa-
   csa-export

   export-
   export-export   # needed for template envvar for CPL saving 

   local path=$(mocknuwa-db)
   mkdir -p $(dirname $path)

   export G4DAECHROMA_DATABASE_PATH=$path
   export G4DAECHROMA_CLIENT_CONFIG=tcp://localhost:5001    # client to local broker

   #export G4DAECHROMA_CLIENT_CONFIG=""
}

mocknuwa--(){
   mocknuwa-make install
   [ $? -ne 0 ] && echo $FUNCNAME failed && return 1
   mocknuwa-run $*
}

mocknuwa-lldb(){
   lldb $(which MockNuWa) $*
}


mocknuwa-run(){
   local msg="=== $FUNCNAME :"

   local evt=$1   # src identifier eg "1"
   shift
   local pfx=$1   # output prefix eg "ha" "hv"
   shift

   local src=$(printf ${DAE_PATH_TEMPLATE} $evt)
   local pho=$(printf ${DAE_PATH_TEMPLATE} $pfx$evt)
   local hit=$(printf ${DAEHIT_PATH_TEMPLATE} $pfx$evt)

   printf " evt %s pfx %s \n" $evt $pfx
   printf " src %s \n" $src 
   printf " pho %s \n" $pho 
   printf " hit %s \n" $hit 

   mocknuwa-runenv 
   #env | grep G4DAECHROMA

   local bin=$(which MockNuWa)
   local cmd="$bin $evt $pfx"
   echo $cmd
   eval $cmd

   ls -l $src $pho $hit 
  

}





