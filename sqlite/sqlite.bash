sqlite-src(){    echo sqlite/sqlite.bash ; }
sqlite-source(){ echo ${BASH_SOURCE:-$(env-home)/$(sqlite-src)} ; }
sqlite-vi(){     vi $(sqlite-source) ; }
sqlitebuild-(){ . $ENV_HOME/sqlite/sqlitebuild/sqlitebuild.bash && sqlitebuild-env $* ; }

sqlite-usage(){ cat << EOU
 
SQLite for Trac
=================

See also sqlite3- for general usage of sqlite


These functions and the precursors: sqlitebuild- and pysqlite-
were prepared with Trac usage in mind. 


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

sqlite-tables(){
  local db=${1:-dybsvn/db/trac.db}
  echo .schema | sqlite3 $db | perl -ne 's,CREATE TABLE (\S*),$1, && print "$1\n"' - 
}


sqlite-notes(){

   cat << EON
   
   prerequisites to trac 
      SQLite, version 3.3.4 and above preferred.
      PySQLite, version 1.x (for SQLite 2.x) or version 2.x (for SQLite 3.x), version 2.3.2 preferred. For details see PySqlite

     i believe pysqlite is included in python 2.5

EON


}


sqlite-again(){

   sqlitebuild-
   sqlitebuild-again
   
   pysqlite-
   pysqlite-again

   apacheconf-
   apacheconf-envvars-add $(sqlite-home)/lib

}

sqlite-name(){ echo sqlite-3.3.16 ; }
sqlite-mode(){ echo ${SQLITE_MODE:-$(sqlite-mode-default $*)} ; }
sqlite-mode-default(){
   case ${1:-$NODE_TAG} in 
        G) echo system ;;
       ZZ) echo source ;;
        *) echo source ;;
   esac
}

sqlite-home(){
   case ${1:-$NODE_TAG} in 
      H) echo $(local-base)/sqlite/$(sqlite-name) ;;
      *) echo $(local-system-base)/sqlite/$(sqlite-name) ;;
   esac
}

sqlite-libdir(){ echo $(sqlite-home)/lib ; }

sqlite-env(){

   elocal-

   [ "$(sqlite-mode)" == "system" ] && return 0
   
   export SQLITE_NAME=$(sqlite-name)
   export SQLITE_HOME=$(sqlite-home)
   
   [ "$NODE_TAG" == "G" ] && return 0
   [ ! -d $SQLITE_HOME ]  && return 0
   
   env-prepend $SQLITE_HOME/bin

   #case $NODE_TAG in
   #  P|XT|C|C2|N) env-llp-prepend $SQLITE_HOME/lib ;;
   #            *) env-ldconfig $SQLITE_HOME/lib    ;;   ##  make available without diddling with llp
   #esac     

   env-llp-prepend $SQLITE_HOME/lib 
}


sqlite-test(){

  # http://trac.edgewall.org/wiki/PySqlite#DetermineactualSQLiteandPySqliteversion
   python -c "import trac.db.sqlite_backend as test ; print test._ver ; print test.have_pysqlite ; print test.sqlite.version ; from pysqlite2 import dbapi2 as sqlite "

}





