# === func-gen- : messaging/cjson/cjs fgp messaging/cjson/cjs.bash fgn cjs fgh messaging/cjson
cjs-src(){      echo messaging/cjson/cjs.bash ; }
cjs-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cjs-src)} ; }
cjs-vi(){       vi $(cjs-source) ; }
cjs-env(){      elocal- ; }
cjs-usage(){ cat << EOU

JSON Usage from C++
===========================


JSON Types
-----------

* http://en.wikipedia.org/wiki/JSON

::

    Number 
          no distinction between float and int
    String
    Boolean
    Array 
    Object
    null

Hmm lack of float/int distinction means need to provide 
type metadata, as wish for ints to be preserved as such 
for selection purposes. 

SQLite3 Types
---------------

* https://www.sqlite.org/datatype3.html

::

    NULL
    INTEGER
    REAL
    TEXT
    BLOB





EOU
}

cjs-name(){  echo cjson ; }
cjs-prefix(){ echo $(cjs-dir) ; }
cjs-libname(){  echo cJSON ; }
cjs-pkgname(){  echo cJSON ; }

cjs-dir(){   echo $(local-base)/env/messaging/$(cjs-name) ; }
cjs-sdir(){ echo $(env-home)/messaging/$(cjs-name) ; }
cjs-bdir(){ echo /tmp/env/messaging/$(cjs-name) ; }

cjs-cd(){  cd $(cjs-sdir); }
cjs-icd(){  cd $(cjs-dir); }
cjs-scd(){  cd $(cjs-sdir)/$1; }
cjs-bcd(){  cd $(cjs-bdir); }


cjs-cmake(){
   type $FUNCNAME
   local iwd=$PWD
   mkdir -p $(cjs-bdir)
   cjs-bcd
   cmake -DCMAKE_INSTALL_PREFIX=$(cjs-prefix) \
         -DCMAKE_BUILD_TYPE=Debug \
         $(cjs-sdir)

   cd $iwd
}
cjs-verbose(){ echo  1 ; }
cjs-make(){
   local iwd=$PWD
   [ ! -d "$(cjs-bdir)" ] && echo needs cjs-build-full first && return 

   cjs-bcd
   make $* VERBOSE=$(cjs-verbose)
   cd $iwd
}
cjs-install(){ cjs-make install ; }

cjs-build(){
   cjs-cmake
   cjs-install
}
cjs--(){ 
   cjs-install 
}

cjs-build-full(){
   cjs-wipe
   cjs-build
}
cjs-wipe(){
   rm -rf $(cjs-bdir)
}


cjs-testcjson(){

   cjs-scd tests

   local nam=testcjson
   local src=$nam.c 
   local bin=$LOCAL_BASE/env/bin/$nam
   local pfx=$(cjs-prefix)
   local lib=$(cjs-libname)

   # bake the place to find the dylib into the binary, so no need for library path
   cc -g $src -I$pfx/include -L$pfx/lib -l$lib -Wl,-rpath,$pfx/lib -o $bin

   $bin $(cjs-sdir)/tests/out.js
}

cjs-testcjson-cmake(){

   local iwd=$PWD
   local tmp=/tmp/env/messaging/cjson/$FUNCNAME
   rm -rf $tmp
   mkdir -p $tmp
   cd $tmp

   cmake -DCMAKE_INSTALL_PREFIX=$(cjs-prefix) \
         -DCMAKE_BUILD_TYPE=Debug \
         $(cjs-sdir)/tests

   make install VERBOSE=1

   cd $iwd 

}


cjs-jstest-make(){
   local pfx=$(cjs-prefix) 
   cjs-scd tests
   clang jstest.cc -lstdc++ -I$pfx/include -L$pfx/lib -lcJSON -Wl,-rpath,$pfx/lib -o $LOCAL_BASE/env/bin/js && js out.js
}





cjs-nuwapkg(){
  if [ -n "$DYB" ]; then
     echo $DYB/NuWa-trunk/dybgaudi/Utilities/$(cjs-pkgname) 
  else
     utilities- && echo $(utilities-dir)/$(cjs-pkgname) 
  fi
}

cjs-nuwapkg-cd(){ cd $(cjs-nuwapkg)/$1 ; }


cjs-names(){
   local path
   local ext=${1:-hh}
   ls -1 $(cjs-sdir)/$(cjs-pkgname)/*.$ext  | while read path ; do
       local name=$(basename $path)
       echo ${name/.$ext}
   done
}


cjs-nuwapkg-action-cmds(){
   local action=${1:-diff}
   local pkg=$(cjs-nuwapkg)
   local pkn=$(basename $pkg)
   local nam=$(cjs-pkgname)
   local sdir=$(cjs-sdir)


   cat << EOC
mkdir -p $pkg/$pkn
mkdir -p $pkg/src
EOC

   local hdxs="hh h"

   local hdx
   local ccx
   for hdx in $hdxs ; do 

      case $hdx in 
         h) ccx=c ;;
        hh) ccx=cc ;;
      esac

   cjs-names $hdx | while read nam ; do

       if [ "$action" == "cpfr" ]; then
         cat << EOC
cp  $pkg/$pkn/$nam.$hdx $sdir/$pkn/$nam.$hdx        
cp  $pkg/src/$nam.$ccx  $sdir/src/$nam.$ccx         
EOC
       else
         cat << EOC
$action $sdir/$pkn/$nam.$hdx  $pkg/$pkn/$nam.$hdx
$action $sdir/src/$nam.$ccx   $pkg/src/$nam.$ccx
EOC
       fi

   done
   done


}

cjs-nuwapkg-action(){
   local cmd
   $FUNCNAME-cmds $1 | while read cmd ; do
      echo $cmd
      eval $cmd
   done
}
cjs-nuwapkg-diff(){  cjs-nuwapkg-action diff ; }
cjs-nuwapkg-cpto(){  cjs-nuwapkg-action cp ; }
cjs-nuwapkg-cpfr(){  cjs-nuwapkg-action cpfr ; }


cjs-nuwapkg-cfg () 
{
    local msg="=== $FUNCNAME :";
    local pkg=$(cjs-nuwapkg);
    shift;
    [ ! -d "$pkg/cmt" ] && echo ERROR NO cmt SUBDIR && sleep 1000000;
    local iwd=$PWD;
    echo $msg for pkg $pkg;
    cd $pkg/cmt;
    cmt config;
    . setup.sh;
    cd $iwd
}

cjs-nuwapkg-build(){
   cjs-nuwapkg-cd
   cjs-nuwapkg-cfg

   cjs-nuwapkg-cd cmt
   cmt br cmt config
   make

}

