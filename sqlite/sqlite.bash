
sqlitebuild-(){ . $ENV_HOME/sqlite/sqlitebuild/sqlitebuild.bash && sqlitebuild-env $* ; }

sqlite-usage(){

   cat << EOU
   
   prerequisites to trac 
      SQLite, version 3.3.4 and above preferred.
      PySQLite, version 1.x (for SQLite 2.x) or version 2.x (for SQLite 3.x), version 2.3.2 preferred. For details see PySqlite

     i believe pysqlite is included in python 2.5

EOU

}

sqlite-env(){

   elocal-
   
   export SQLITE_NAME=sqlite-3.3.16
   export SQLITE_HOME=$LOCAL_BASE/sqlite/$SQLITE_NAME
   
   #export LD_LIBRARY_PATH=$SQLITE_HOME/lib:$LD_LIBRARY_PATH
    
   [ "$NODE_TAG" == "G" ] && return 0
   sqlite-path
   
}

sqlite-path(){

  local msg="=== $FUNCNAME :"
  [ -z $SQLITE_HOME ] && echo $msg skipping as no SQLITE_HOME && return 1  
  env-prepend $SQLITE_HOME/bin
  
}




