sqlitebuild-src(){    echo sqlite/sqlitebuild/sqlitebuild.bash ; }
sqlitebuild-source(){ echo ${BASH_SOURCE:-$(env-home)/$(sqlitebuild-src)} ; }
sqlitebuild-vi(){     vi $(sqlitebuild-source) ; }
sqlitebuild-usage(){ cat << EOU

SQLite Building for Trac
=========================

   
sqlitebuild-dir   :   $(sqlitebuild-dir) 
   
sqlitebuild-get/configure/install/wipe/wipe-install
      
sqlitebuild-copy-to-node  <node-tag>
              copy the tarball to another node

sqlitebuild-copy-from-node  <node-tag>
              copy the tarball from another node

sqlitebuild-tgz-from-backup tgz
              extract from the heprez backup tarball
              sqlite source tarball distribution has changed a lot,
              a brief look does not yield the old tarballs... so grab from backup


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
sqlitebuild-htdocs-url(){  echo $(TRAC_INSTANCE=heprez htdocs-url)/$(sqlitebuild-tgz) ; }

sqlitebuild-fold(){ echo $(local-system-base)/sqlite ; }

sqlitebuild-upload(){
   local msg="=== $FUNCNAME: "
   htdocs-
   local path=$(sqlitebuild-fold)/$(sqlitebuild-tgz)
   [ ! -f $path ] && echo $msg ERROR no $path && return 1
   TRAC_INSTANCE=heprez htdocs-up $path 
}

sqlitebuild-tgz-from-backup(){
    local msg="=== $FUNCNAME :"
    local tgz=$1
    local chktgz=sqlite-3.3.16.tar.gz
    [ "$tgz" != "$chktgz" ] && echo $msg tgz is not the traditional one in backups && return 1

    local bkp=$SCM_FOLD/backup/cms02/tracs/heprez/last/heprez.tar.gz 
    local rpath=heprez/htdocs/$tgz 

    if [ -f "$bkp" ]; then
        if [ ! -f "$tgz" ]; then  
            echo $msg extract rpath $rpath from bkp $bkp to $PWD/$tgz
            tar Ozxf $bkp $rpath > $tgz
        else
            echo $msg tgz present already $tgz  
        fi 
    fi
}

sqlitebuild-get(){
 
    local msg="=== $FUNCNAME :"
    local nam=$SQLITE_NAME
    local tgz=$(sqlitebuild-tgz)

    local dir=$(sqlitebuild-fold)
    cd $(dirname $dir)
    mkdir -p $(basename $dir)
    cd $dir 

    # this is useless when rebuilding following server death, as requires the server to be alive
    # [ ! -f $tgz ] && env-mcurl $(sqlitebuild-url) $(sqlitebuild-htdocs-url)

    sqlitebuild-tgz-from-backup $tgz
    
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

sqlitebuild-check(){
    echo -n
}

sqlitebuild-again(){

  sqlitebuild-wipe
  sqlitebuild-wipe-install
  
  sqlitebuild-get
  sqlitebuild-configure
  sqlitebuild-install

  sqlitebuild-check
}



