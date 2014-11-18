# === func-gen- : messaging/cjson/cjs fgp messaging/cjson/cjs.bash fgn cjs fgh messaging/cjson
cjs-src(){      echo messaging/cjson/cjs.bash ; }
cjs-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cjs-src)} ; }
cjs-vi(){       vi $(cjs-source) ; }
cjs-env(){      elocal- ; }
cjs-usage(){ cat << EOU





EOU
}
cjs-dir(){ echo $(local-base)/env/messaging/cjs ; }
cjs-cd(){  cd $(cjs-dir); }
cjs-mate(){ mate $(cjs-dir) ; }
cjs-get(){
   local dir=$(dirname $(cjs-dir)) &&  mkdir -p $dir && cd $dir

}

cjs-name(){  echo cJSON ; }
cjs-dir(){   echo $(local-base)/env/messaging/$(cjs-name) ; }
cjs-prefix(){ echo $(cjs-dir) ; }

cjs-sdir(){ echo $(env-home)/chroma/$(cjs-name) ; }
cjs-bdir(){ echo /tmp/env/chroma/$(cjs-name) ; }

cjs-cd(){  cd $(cjs-sdir); }
cjs-icd(){  cd $(cjs-dir); }
cjs-scd(){  cd $(cjs-sdir); }
cjs-bcd(){  cd $(cjs-bdir); }



cjs-cmake(){
   type $FUNCNAME
   local iwd=$PWD
   mkdir -p $(cjs-bdir)
   cjs-bcd
   cmake -DGeant4_DIR=$(cjs-geant4-dir) \
         -DCMAKE_INSTALL_PREFIX=$(cjs-prefix) \
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
   #cjs-make
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



