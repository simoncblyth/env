
sqlitebuild-usage(){
  
   cat << EOU

   


EOU

}

sqlitebuild-env(){
   sqlite-
}

sqlitebuild-get(){
 
    local nam=$SQLITE_NAME
    local tgz=$nam.tar.gz
    local url=http://www.sqlite.org/$tgz

    cd $SYSTEM_BASE
    mkdir -p sqlite
    cd sqlite

    [ ! -f $tgz ] && curl -O $url
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




sqlitebuild-dir(){
   case $NODE_TAG in 
      H) echo $(local-base)/sqlite/build/$(sqlite-name) ;;
      *) echo $(system-base)/sqlite/build/$(sqlite-name) ;;
   esac
}


sqlitebuild-configure(){

  cd $(sqlitebuild-dir)
  ./configure -h 
  ./configure --prefix=$SQLITE_HOME --disable-tcl
  
}

sqlitebuild-install(){
   
   cd $(sqlitebuild-dir)
   make 
   make install

}

sqlitebuild-wipe(){
   local iwd=$PWD
   cd $SYSTEM_BASE/sqlite
   [ -d build ] && rm -rf build
   cd $iwd
}

sqlitebuild-wipe-install(){
   local iwd=$PWD
   cd $SYSTEM_BASE/sqlite
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
