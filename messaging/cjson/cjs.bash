# === func-gen- : messaging/cjson/cjs fgp messaging/cjson/cjs.bash fgn cjs fgh messaging/cjson
cjs-src(){      echo messaging/cjson/cjs.bash ; }
cjs-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cjs-src)} ; }
cjs-vi(){       vi $(cjs-source) ; }
cjs-env(){      elocal- ; }
cjs-usage(){ cat << EOU





EOU
}

cjs-name(){  echo cjson ; }
cjs-prefix(){ echo $(cjs-dir) ; }

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


cjs-test(){

   cjs-scd tests

   local nam=testcjson
   local src=$nam.c 
   local bin=$LOCAL_BASE/env/bin/$nam
   local pfx=$(cjs-prefix)
   local lib=cJSON

   # bake the place to find the dylib into the binary, so no need for library path
   cc -g $src -I$pfx/include -L$pfx/lib -l$lib -Wl,-rpath,$pfx/lib -o $bin

   $bin $(cjs-sdir)/tests/out.js
}

cjs-test-cmake(){

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

