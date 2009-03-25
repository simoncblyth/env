sqlitebuild-src(){    echo sqlite/sqlitebuild/sqlitebuild.bash ; }
sqlitebuild-source(){ echo ${BASH_SOURCE:-$(env-home)/$(sqlitebuild-src)} ; }
sqlitebuild-vi(){     vi $(sqlitebuild-source) ; }
sqlitebuild-usage(){
  
   cat << EOU

   
      sqlitebuild-dir   :   $(sqlitebuild-dir) 
   
      sqlitebuild-get/configure/install/wipe/wipe-install
      
      sqlitebuild-copy-to-node  <node-tag>
              copy the tarball to another node

      sqlitebuild-copy-from-node  <node-tag>
              copy the tarball from another node


      $(type sqlitebuild-again)



EOU

}

sqlitebuild-env(){
   sqlite-
}


sqlitebuild-urlbase(){
   case $NODE_TAG in
     XT|XX) echo http://www.sqlite.com.cn/Upfiles/sqlite ;;
         *) echo http://www.sqlite.com ;; 
   esac
}  

sqlitebuild-tgz(){ echo $SQLITE_NAME.tar.gz ; }
sqlitebuild-url(){ echo $(sqlitebuild-urlbase)/$(sqlitebuild-tgz) ; }
sqlitebuild-get(){
 
    local msg="=== $FUNCNAME :"
    local nam=$SQLITE_NAME
    local tgz=$(sqlitebuild-tgz)

    cd $SYSTEM_BASE
    mkdir -p sqlite
    cd sqlite

    [ ! -f $tgz ] && env-curl $(sqlitebuild-url)
    
    file-
    file-size-lt $tgz 100 && echo $msg ABORT tgz $tgz is too small ... sleeping && sleep 10000000000
    
    mkdir -p  build
    [ ! -d build/$nam ] && tar -C build -zxvf $tgz 
}


sqlitebuild-copy-to-node(){

   local msg="=== $FUNCNAME :"
   local t=${1:-C}
   local cmd="scp $(sqlite-home).tar.gz $t:$(NODE_TAG=$t sqlite-home).tar.gz"
   echo $msg $cmd
   eval $cmd

}

sqlitebuild-copy-from-node(){

   local msg="=== $FUNCNAME :"
   local t=${1:-C}
   local cmd="scp  $t:$(NODE_TAG=$t sqlite-home).tar.gz $(sqlite-home).tar.gz"
   echo $msg $cmd
   eval $cmd

}





sqlitebuild-dir(){
   case $NODE_TAG in 
      H) echo $(local-base)/sqlite/build/$(sqlite-name) ;;
      *) echo $(local-system-base)/sqlite/build/$(sqlite-name) ;;
   esac
}


sqlitebuild-configure(){

  cd $(sqlitebuild-dir)
  ./configure -h 
  ./configure --prefix=$(sqlite-home) --disable-tcl
  
}

sqlitebuild-install(){
   
   cd $(sqlitebuild-dir)
   make 
   make install

}

sqlitebuild-wipe(){
   local iwd=$PWD
   local dir=$SYSTEM_BASE/sqlite
   [ ! -d $dir ] && return 0
   cd $dir
   [ -d build ] && rm -rf build
   cd $iwd
}

sqlitebuild-wipe-install(){
   local iwd=$PWD
   local dir=$SYSTEM_BASE/sqlite
   [ ! -d $dir ] && return 0
   cd $dir
   [ "${SQLITE_NAME:0:6}" != "sqlite" ] && echo bad name $SQLITE_NAME cannot proceed && return 1
   [ -d $SQLITE_NAME ] && rm -rf $SQLITE_NAME
   cd $iwd
}


sqlitebuild-again(){

  sqlitebuild-wipe
  sqlitebuild-wipe-install
  
  sqlitebuild-get
  sqlitebuild-configure
  sqlitebuild-install

  sqlitebuild-check
}



sqlitebuild-check(){
    echo -n
}
