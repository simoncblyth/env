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


::

    2014-11-20 19:47:44,927 INFO    env.geant4.geometry.collada.g4daeview.daechromacontext:66  using seed 0 
    2014-11-20 19:47:44,927 INFO    env.geant4.geometry.collada.g4daeview.g4daechroma:73  ***** post G4DAEChroma ctor
    2014-11-20 19:47:44,927 INFO    env.geant4.geometry.collada.g4daeview.g4daechroma:50  start polling responder: DAEDirectResponder connect tcp://127.0.0.1:5002  
    2014-11-20 19:47:44,927 INFO    env.geant4.geometry.collada.g4daeview.g4daechroma:54  polling 0 
    2014-11-20 19:47:52,657 INFO    env.geant4.geometry.collada.g4daeview.daedirectresponder:47  DAEDirectResponder request (4165, 4, 4) 
    2014-11-20 19:47:52,657 INFO    env.geant4.geometry.collada.g4daeview.g4daechroma:43  handler got obj (cpl or npl)
    2014-11-20 19:47:52,657 INFO    env.geant4.geometry.collada.g4daeview.daedirectpropagator:54  ctrl {u'reset_rng_states': 1, u'nthreads_per_block': 64, u'seed': 0, u'max_blocks': 1024, u'max_steps': 30, u'COLUMNS': u'max_blocks:i,max_steps:i,nthreads_per_block:i,reset_rng_states:i,seed:i'} 
    2014-11-20 19:47:52,657 WARNING env.geant4.geometry.collada.g4daeview.daedirectpropagator:62  reset_rng_states
    2014-11-20 19:47:52,657 INFO    env.geant4.geometry.collada.g4daeview.daechromacontext:122 _set_rng_states
    2014-11-20 19:47:53,005 INFO    chroma.gpu.geometry :171 Splitting BVH between GPU and CPU memory at node 100
    2014-11-20 19:47:53,030 INFO    chroma.gpu.geometry :201 device usage:
    ----------
    nodes           100.0    1.6K
    total                    1.6K
    ----------
    device total             2.1G
    device used              1.9G
    device free            292.1M

    2014-11-20 19:47:53,033 INFO    env.geant4.geometry.collada.g4daeview.daechromacontext:117 _get_rng_states
    2014-11-20 19:47:53,033 INFO    env.geant4.geometry.collada.g4daeview.daechromacontext:72  setup_rng_states using seed 0 





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
   mocknuwa-runenv 
   lldb $(which MockNuWa) $*
}


mocknuwa-run(){
   local msg="=== $FUNCNAME :"


   local evt=$1   # src identifier eg "1"
   shift
   local pfx=$1   # output prefix eg "ha" "hv"
   shift


   if [ "$evt" != "MOCK" ]; then 
       local src=$(printf ${DAE_PATH_TEMPLATE} $evt)
       local pho=$(printf ${DAE_PATH_TEMPLATE} $pfx$evt)
       local hit=$(printf ${DAEHIT_PATH_TEMPLATE} $pfx$evt)

       printf " evt %s pfx %s \n" $evt $pfx
       printf " src %s \n" $src 
       printf " pho %s \n" $pho 
       printf " hit %s \n" $hit 
   fi

   mocknuwa-runenv 
   #env | grep G4DAECHROMA

   local bin=$(which MockNuWa)
   local cmd="$bin $evt $pfx"
   echo $cmd
   eval $cmd

   [ "$evt" != "MOCK" ] && ls -l $src $pho $hit 
  

}





