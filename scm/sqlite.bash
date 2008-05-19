#
#
#  sqlite-x
#  sqlite-i
#  
#  sqlite-get
#  sqlite-wipe
#  sqlite-install
#
#
#
#  prerequisites to trac 
#    SQLite, version 3.3.4 and above preferred.
#    PySQLite, version 1.x (for SQLite 2.x) or version 2.x (for SQLite 3.x), version 2.3.2 preferred. For details see PySqlite
#
#  prerequisites to PySQLite :  python 
#  see python.bash
#
#

sqlite-env(){

   elocal-
   sqlite-base
   
   if [ "$NODE_TAG" != "G" ]; then
      sqlite-path
   fi
}


sqlite-base(){

   local SQLITE_NAME=sqlite-3.3.16
   export SQLITE_HOME=$LOCAL_BASE/sqlite/$SQLITE_NAME
   #export LD_LIBRARY_PATH=$SQLITE_HOME/lib:$LD_LIBRARY_PATH
}


sqlite-path(){

  local msg="=== $FUNCNAME :"
  [ -z $SQLITE_HOME ] && echo $msg skipping as no SQLITE_HOME && return 1
  export PATH=$SQLITE_HOME/bin:$PATH

}



sqlite-get(){

  [ "$NODE" == "g4pb" ] && echo use the patched version of sqlite supplied  with OSX && return

  local nam=$SQLITE_NAME
  local tgz=$nam.tar.gz
  local url=http://www.sqlite.org/$tgz

  cd $LOCAL_BASE
  test -d sqlite || ( $SUDO mkdir sqlite && $SUDO chown $USER sqlite )
  cd sqlite 

  test -f $tgz || curl -o $tgz $url
  test -d build || mkdir build
  test -d build/$nam || tar -C build -zxvf $tgz 
}

sqlite-wipe(){

  local nam=$SQLITE_NAME
  cd $LOCAL_BASE/sqlite
  rm -rf build/$nam

}

sqlite-install(){

  local nam=$SQLITE_NAME
  cd $LOCAL_BASE/sqlite/build/$nam

  ./configure -h 
  ./configure --prefix=$LOCAL_BASE/sqlite/$nam --disable-tcl
  make   
  make install  
}


