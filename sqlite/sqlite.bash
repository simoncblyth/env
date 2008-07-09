
sqlitebuild-(){ . $ENV_HOME/sqlite/sqlitebuild/sqlitebuild.bash && sqlitebuild-env $* ; }
pysqlite-(){    . $ENV_HOME/sqlite/sqlitebuild/pysqlite.bash    && pysqlite-env $* ; }

sqlite-usage(){

   cat << EOU
   
   prerequisites to trac 
      SQLite, version 3.3.4 and above preferred.
      PySQLite, version 1.x (for SQLite 2.x) or version 2.x (for SQLite 3.x), version 2.3.2 preferred. For details see PySqlite

     i believe pysqlite is included in python 2.5


            sqlite-name :  $(sqlite-name)
            sqlite-home :  $(sqlite-home)
            

            sqlite-ldconfig :
                   make available without diddling with llp


EOU

}


sqlite-name(){
   echo sqlite-3.3.16
}

sqlite-home(){
   case $NODE_TAG in 
      H) echo $(local-base)/sqlite/$(sqlite-name) ;;
      *) echo $(system-base)/sqlite/$(sqlite-name) ;;
   esac
}

sqlite-env(){

   elocal-
   
   export SQLITE_NAME=$(sqlite-name)
   export SQLITE_HOME=$(sqlite-home)
   
   #export LD_LIBRARY_PATH=$SQLITE_HOME/lib:$LD_LIBRARY_PATH
    
   [ "$NODE_TAG" == "G" ] && return 0
   
   sqlite-path
   
}

sqlite-path(){

  local msg="=== $FUNCNAME :"
  [ -z $SQLITE_HOME ] && echo $msg skipping as no SQLITE_HOME && return 1  
  
  env-prepend $SQLITE_HOME/bin
  
  
}


sqlite-ldconfig(){

  env-ldconfig $(sqlite-home)/lib 

}

sqlite-test(){

  # http://trac.edgewall.org/wiki/PySqlite#DetermineactualSQLiteandPySqliteversion
   python -c "import trac.db.sqlite_backend as test ; print test._ver ; print test.have_pysqlite ; print test.sqlite.version ; from pysqlite2 import dbapi2 as sqlite "

}





