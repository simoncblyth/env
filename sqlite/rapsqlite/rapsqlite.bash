# === func-gen- : sqlite/rapsqlite/rapsqlite fgp sqlite/rapsqlite/rapsqlite.bash fgn rapsqlite fgh sqlite/rapsqlite
rapsqlite-src(){      echo sqlite/rapsqlite/rapsqlite.bash ; }
rapsqlite-source(){   echo ${BASH_SOURCE:-$(env-home)/$(rapsqlite-src)} ; }
rapsqlite-vi(){       vi $(rapsqlite-source) ; }
rapsqlite-env(){      elocal- ; }
rapsqlite-usage(){ cat << EOU

RapSqlite : Yet Another SQLite3 C++ Wrapper  
=============================================


Related
---------

* sqlite- for Trac usage 
* sqlite3- building 


SQLite3 Refs
---------------

* https://www.sqlite.org/autoinc.html

  * Auto increment Primary Key 
  * NB there is an invisible ROWID/OID column in all sqlite tables
    but better to make it explicit (in case ever need to move 
    to another DB) via an "id integer primary key" 

* http://www.sqlite.org/backup.html

  * copy memory db to file

* http://beets.radbox.org/blog/sqlite-nightmare.html

  * concurrency, locking issues


SQLite3 Swift Wrapper
----------------------

* https://github.com/stephencelis/SQLite.swift/blob/master/Documentation/Index.md



Install as NuWa Utility
-------------------------

Depends in $DYB/NuWa-trunk/lcgcmt/LCG_Builders/sqlite/cmt/requirements 

::

	./dybinst trunk external sqlite





EOU
}
rapsqlite-dir(){ echo $(local-base)/env/sqlite/rapsqlite ; }
rapsqlite-prefix(){ echo $(rapsqlite-dir) ; }
rapsqlite-name(){ echo RapSqlite ; }

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
   local dbp=$LOCAL_BASE/env/nuwa/mocknuwa.db
   local tab="mocknuwa"

   # bake the place to find the dylib into the binary, so no need for library path
   clang $src -lstdc++ \
               -I$pfx/include \
               -L$pfx/lib -l$lib \
               -Wl,-rpath,$pfx/lib \
               -o $bin \
        && DBPATH=$dbp $LLDB $nam \
        && echo select \* from $tab \; | sqlite3 $dbp
}

rapsqlite-lldb(){
   LLDB=lldb rapsqlite-test
}



rapsqlite-lib(){ echo RapSqlite ; }

rapsqlite-testjs(){

   cjs- 
   rapsqlite-scd tests

   local nam=testrapsqlitejs
   local dbp=$LOCAL_BASE/env/nuwa/mocknuwa.db

   clang $nam.cc -lstdc++ \
              -I$(rapsqlite-prefix)/include \
              -I$(cjs-prefix)/include \
              -L$(rapsqlite-prefix)/lib -l$(rapsqlite-lib) \
              -L$(cjs-prefix)/lib -l$(cjs-lib) \
              -Wl,-rpath,$(rapsqlite-prefix)/lib \
              -Wl,-rpath,$(cjs-prefix)/lib \
              -o $LOCAL_BASE/env/bin/$nam \
        && DBPATH=$dbp $LLDB $nam

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



rapsqlite-nuwapkg(){    
  if [ -n "$DYB" ]; then 
     echo $DYB/NuWa-trunk/dybgaudi/Utilities/$(rapsqlite-name) 
  else
     utilities- && echo $(utilities-dir)/$(rapsqlite-name) 
  fi
}

rapsqlite-nuwapkg-cd(){ cd $(rapsqlite-nuwapkg)/$1 ; } 


rapsqlite-names(){ 
   local path
   ls -1 $(rapsqlite-sdir)/$(rapsqlite-name)/*.hh  | while read path ; do
      local name=$(basename $path)
      echo ${name/.hh}
   done
}




rapsqlite-nuwapkg-action-cmds(){
   local action=${1:-diff}
   local pkg=$(rapsqlite-nuwapkg)
   local pkn=$(basename $pkg)
   local nam=$(rapsqlite-name)

   cat << EOC
mkdir -p $pkg/$pkn
mkdir -p $pkg/src
EOC

   rapsqlite-names |while read nam ; do
   cat << EOC
$action $(rapsqlite-sdir)/$pkn/$nam.hh         $pkg/$pkn/$nam.hh
$action $(rapsqlite-sdir)/src/$nam.cc          $pkg/src/$nam.cc
EOC
   done

}

rapsqlite-nuwapkg-action(){
   local cmd
   $FUNCNAME-cmds $1 | while read cmd ; do
      echo $cmd
      eval $cmd
   done
}
rapsqlite-nuwapkg-diff(){  rapsqlite-nuwapkg-action diff ; }
rapsqlite-nuwapkg-cpto(){  rapsqlite-nuwapkg-action cp ; }

rapsqlite-nuwacfg () 
{ 
    local msg="=== $FUNCNAME :";
    local pkg=$(rapsqlite-nuwapkg);
    shift;
    [ ! -d "$pkg/cmt" ] && echo ERROR NO cmt SUBDIR && sleep 1000000;
    local iwd=$PWD;
    echo $msg for pkg $pkg;
    cd $pkg/cmt;
    cmt config;
    . setup.sh;
    cd $iwd
}
