
swishbuild-usage(){
  
   cat << EOU

   
      swishbuild-dir   :   $(swishbuild-dir) 
   
      swishbuild-get/configure/install/wipe/wipe-install
      
      swishbuild-copy-to-node  <node-tag>
              copy the tarball to another node

      swishbuild-copy-from-node  <node-tag>
              copy the tarball from another node


      $(type swishbuild-again)



EOU

}

swishbuild-env(){
   swish-
}


swishbuild-urlbase(){
     echo http://swish-e.org/distribution
}  

swishbuild-get(){
 
    local msg="=== $FUNCNAME :"
    local nam=$SWISH_NAME
    local tgz=$nam.tar.gz
    local url=$(swishbuild-urlbase)/$tgz

    cd $SYSTEM_BASE
    mkdir -p swish
    cd swish

    [ ! -f $tgz ] && curl -O $url
    
    file-
    file-size-lt $tgz 100 && echo $msg ABORT tgz $tgz is too small ... sleeping && sleep 10000000000
    
    mkdir -p  build
    [ ! -d build/$nam ] && tar -C build -zxvf $tgz 
}


swishbuild-copy-to-node(){

   local msg="=== $FUNCNAME :"
   local t=${1:-C}
   local cmd="scp $(swish-home).tar.gz $t:$(NODE_TAG=$t swish-home).tar.gz"
   echo $msg $cmd
   eval $cmd

}

swishbuild-copy-from-node(){

   local msg="=== $FUNCNAME :"
   local t=${1:-C}
   local cmd="scp  $t:$(NODE_TAG=$t swish-home).tar.gz $(swish-home).tar.gz"
   echo $msg $cmd
   eval $cmd

}





swishbuild-dir(){
   case $NODE_TAG in 
      H) echo $(local-base)/swish/build/$(swish-name) ;;
      *) echo $(local-system-base)/swish/build/$(swish-name) ;;
   esac
}


swishbuild-configure(){

  cd $(swishbuild-dir)
  ./configure -h 
  ./configure --prefix=$(swish-home) --enable-incremental
  
}

swishbuild-install(){
   
   cd $(swishbuild-dir)
   make 
   make install

}

swishbuild-wipe(){
   local iwd=$PWD
   local dir=$SYSTEM_BASE/swish
   [ ! -d $dir ] && return 0
   cd $dir
   [ -d build ] && rm -rf build
   cd $iwd
}

swishbuild-wipe-install(){
   local iwd=$PWD
   local dir=$SYSTEM_BASE/swish
   [ ! -d $dir ] && return 0
   cd $dir
   [ "${SWISH_NAME:0:6}" != "swish" ] && echo bad name $SWISH_NAME cannot proceed && return 1
   [ -d $SWISH_NAME ] && rm -rf $SWISH_NAME
   cd $iwd
}


swishbuild-again(){

  swishbuild-wipe
  swishbuild-wipe-install
  
  swishbuild-get
  swishbuild-configure
  swishbuild-install

  swishbuild-check
}



swishbuild-check(){
    echo -n
}
