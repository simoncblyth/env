
sqlitebuild-(){ . $ENV_HOME/sqlite/sqlitebuild/sqlitebuild.bash && sqlitebuild-env $* ; }
pysqlite-(){    . $ENV_HOME/sqlite/sqlitebuild/pysqlite.bash    && pysqlite-env $* ; }

sqlite-usage(){

   cat << EOU
 
     sqlite-name   :  $(sqlite-name)
     sqlite-home   :  $(sqlite-home)
     which sqlite3 :  $(which sqlite3)       
            
     sqlite-env :
                   invoked by precursor
                   sets up the PATH and LD_LIBRARY_PATH or ldconfig

     $(type sqlite-again)

     sqlite-test 
     
                   
     Precursors...
            
     sqlitebuild-
     pysqlite-



EOU

}



sqlite-notes(){

   cat << EOU
   
   prerequisites to trac 
      SQLite, version 3.3.4 and above preferred.
      PySQLite, version 1.x (for SQLite 2.x) or version 2.x (for SQLite 3.x), version 2.3.2 preferred. For details see PySqlite

     i believe pysqlite is included in python 2.5

EOU


}


sqlite-again(){

   sqlitebuild-
   sqlitebuild-again
   
   pysqlite-
   pysqlite-again

}

sqlite-name(){
   echo sqlite-3.3.16
}

sqlite-home(){
   case ${1:-$NODE_TAG} in 
      H) echo $(local-base)/sqlite/$(sqlite-name) ;;
      *) echo $(local-system-base)/sqlite/$(sqlite-name) ;;
   esac
}

sqlite-env(){

   elocal-
   
   export SQLITE_NAME=$(sqlite-name)
   export SQLITE_HOME=$(sqlite-home)
   
   [ "$NODE_TAG" == "G" ] && return 0
   [ ! -d $SQLITE_HOME ]  && return 0
   
   env-prepend $SQLITE_HOME/bin

   case $NODE_TAG in
     P|XT) env-llp-prepend $SQLITE_HOME/lib ;;
        *) env-ldconfig $SQLITE_HOME/lib    ;;   ##  make available without diddling with llp
   esac     
}


sqlite-test(){

  # http://trac.edgewall.org/wiki/PySqlite#DetermineactualSQLiteandPySqliteversion
   python -c "import trac.db.sqlite_backend as test ; print test._ver ; print test.have_pysqlite ; print test.sqlite.version ; from pysqlite2 import dbapi2 as sqlite "

}





