# === func-gen- : sqlite/rapsqlite/rapsqlite fgp sqlite/rapsqlite/rapsqlite.bash fgn rapsqlite fgh sqlite/rapsqlite
rapsqlite-src(){      echo sqlite/rapsqlite/rapsqlite.bash ; }
rapsqlite-source(){   echo ${BASH_SOURCE:-$(env-home)/$(rapsqlite-src)} ; }
rapsqlite-vi(){       vi $(rapsqlite-source) ; }
rapsqlite-env(){      elocal- ; }
rapsqlite-usage(){ cat << EOU





EOU
}
rapsqlite-dir(){ echo $(local-base)/env/sqlite/rapsqlite ; }
rapsqlite-prefix(){ echo $(rapsqlite-dir) ; }

rapsqlite-sdir(){ echo $(env-home)/sqlite/rapsqlite ; }
rapsqlite-bdir(){ echo /tmp/env/sqlite/rapsqlite ; }

rapsqlite-cd(){   cd $(rapsqlite-dir); }
rapsqlite-scd(){  cd $(rapsqlite-sdir)/$1; }
rapsqlite-bcd(){  cd $(rapsqlite-bdir); }

rapsqlite-verbose(){  echo 1 ; }

rapsqlite-cmake(){
   type $FUNCNAME
   local iwd=$PWD
   mkdir -p $(rapsqlite-bdir)
   rapsqlite-bcd

   cmake -DCMAKE_INSTALL_PREFIX=$(rapsqlite-prefix) \
         -DCMAKE_BUILD_TYPE=Debug \
         $(rapsqlite-sdir)

   cd $iwd
}
rapsqlite-verbose(){ echo  1 ; }
rapsqlite-make(){
   local iwd=$PWD
   rapsqlite-bcd
   make $* VERBOSE=$(rapsqlite-verbose)
   cd $iwd
}
rapsqlite-install(){ rapsqlite-make install ; }

rapsqlite-build(){
   rapsqlite-cmake
   rapsqlite-install
}
rapsqlite--(){ 
   rapsqlite-install 
}

rapsqlite-build-full(){
   rapsqlite-wipe
   rapsqlite-build
}
rapsqlite-wipe(){
   rm -rf $(rapsqlite-bdir)
}


rapsqlite-rpath(){
   otool-;otool-rpath $(rapsqlite-prefix)/lib/libRapSqlite.dylib
}


rapsqlite-test(){

   rapsqlite-scd tests

   local nam=testrapsqlite
   local src=$nam.cc 
   local bin=$LOCAL_BASE/env/bin/$nam
   local pfx=$(rapsqlite-prefix)
   local lib=RapSqlite
   local dbp=/tmp/rap.db

   # bake the place to find the dylib into the binary, so no need for library path
   clang $src -lstdc++ -I$pfx/include -L$pfx/lib -l$lib -Wl,-rpath,$pfx/lib -o $bin

   DBPATH=$dbp $nam
   echo select \* from D \; | sqlite3 $dbp
 
}

rapsqlite-test-cmake(){

   local iwd=$PWD
   local tmp=/tmp/env/sqlite/$FUNCNAME
   rm -rf $tmp
   mkdir -p $tmp
   cd $tmp

   cmake -DCMAKE_INSTALL_PREFIX=$(rapsqlite-prefix) \
         -DCMAKE_BUILD_TYPE=Debug \
         $(rapsqlite-sdir)/tests

   make install VERBOSE=1

   cd $iwd 

}
