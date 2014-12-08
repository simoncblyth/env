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

mocknuwa-export(){
   export SQLITE3_DATABASE=$(mocknuwa-db) 
}


mocknuwa-usage(){ cat << EOU

MockNuWa
=========

ipython.sh::

    In [4]: a = q("select oid, nwork, tottime from test ;")
    npar: envvar SQLITE3_DATABASE:/usr/local/env/nuwa/mocknuwa.db ncol 3 nrow 48 type f  fbufmax 1000  

    In [5]: a
    Out[5]: 
    array([[    1.   ,   445.   ,     0.07 ],
           [    2.   ,    24.   ,     0.013],
           [    3.   ,  1888.   ,     0.222],

    In [13]: plt.scatter(a[:,1],a[:,2])
    Out[13]: <matplotlib.collections.PathCollection at 0x1057bbe50>

    In [14]: plt.show()




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
   mocknuwa-runenv 
   MockNuWa $*   
}

mocknuwa-npy(){
   local base=$(dirname $DAE_PATH_TEMPLATE)
   local path
   local name
   local evt
   for path in $(ls -1 $base/2014*.npy) ; do
       name=${path/$base\/}
       evt=${name/.npy/}
       echo $evt
   done
}

mocknuwa-all(){
  local tag
  mocknuwa-npy | while read tag ; do
     echo $tag
     mocknuwa-run $tag pp
  done
}


mocknuwa-tableinfo(){
   echo pragma table_info\(mocknuwa\)\; | sqlite3 -cmd '.header ON' -cmd '.width 5 20 20 5 5' -column $(mocknuwa-db)
}

mocknuwa-ctrl-setup(){ 
   mocknuwa-export
   python $(mocknuwa-sdir)/ctrl.py $* ; 
}
mocknuwa-ctrl(){  mocknuwa-query ctrl; }
mocknuwa-batch(){ mocknuwa-query batch; }
mocknuwa-query(){ echo select \* from ${1:-ctrl} \; |  sqlite3 -cmd '.header ON' -column $(mocknuwa-db) ; }


mocknuwa-scan(){
   mocknuwa-runenv 
   MockNuWa 1:49 1:17
}

mocknuwa-scanone(){
   mocknuwa-runenv 
   MockNuWa 1:2 1:17
}

mocknuwa-log-drop(){
   echo drop table if exists log \; | mocknuwa-sqlite
}

mocknuwa-log(){
   echo select \* from log \; | mocknuwa-sqlite
}

