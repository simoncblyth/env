# === func-gen- : sqlite/testsqlite/testsqlite fgp sqlite/testsqlite/testsqlite.bash fgn testsqlite fgh sqlite/testsqlite
testsqlite-src(){      echo sqlite/testsqlite/testsqlite.bash ; }
testsqlite-source(){   echo ${BASH_SOURCE:-$(env-home)/$(testsqlite-src)} ; }
testsqlite-vi(){       vi $(testsqlite-source) ; }
testsqlite-env(){      elocal- ; }
testsqlite-usage(){ cat << EOU

Test SQLite C API
===================

Reflection
-----------

* http://stackoverflow.com/questions/604939/how-can-i-get-the-list-of-a-columns-in-a-table-for-a-sqlite-database

::

    delta:~ blyth$ testsqlite-tableinfo
    0|type|text|0||0
    1|name|text|0||0
    2|tbl_name|text|0||0
    3|rootpage|integer|0||0
    4|sql|text|0||0

    delta:~ blyth$ testsqlite-tableinfo A
    0|x|int|0||0
    1|y|string|0||0

    delta:~ blyth$ testsqlite-tableinfo COMPANY
    0|ID|INT|1||1
    1|NAME|TEXT|1||0
    2|AGE|INT|1||0
    3|ADDRESS|CHAR(50)|0||0
    4|SALARY|REAL|0||0



v3 stepping API rather than callback
-------------------------------------

* https://www.sqlite.org/capi3.html






EOU
}
testsqlite-dir(){ echo $(local-base)/env/sqlite/testsqlite ; }
testsqlite-sdir(){ echo $(env-home)/sqlite/testsqlite ; }
testsqlite-cd(){  cd $(testsqlite-dir); }
testsqlite-scd(){  cd $(testsqlite-sdir); }
testsqlite-mate(){ mate $(testsqlite-dir) ; }
testsqlite-get(){
   local dir=$(dirname $(testsqlite-dir)) &&  mkdir -p $dir && cd $dir

}

testsqlite-db(){ echo /tmp/testsqlite.db ; }


testsqlite-tableinfo(){
    local table=${1:-sqlite_master}
    echo pragma table_info\($table\)\; | sqlite3 $(testsqlite-db)
}

testsqlite-cli(){
     sqlite3 $(testsqlite-db)
}

testsqlite--()
{

   local src=$(testsqlite-sdir)/testsqlite.cc
   local bin=$LOCAL_BASE/env/bin/testsqlite

   #cc $src -L/opt/local/lib -lsqlite3 -o $bin && testsqlite $(testsqlite-db)
   clang $src -L/opt/local/lib -lsqlite3 -lstdc++  -o $bin && DBPATH=$(testsqlite-db) testsqlite

   #echo select \* from A \; | sqlite3 $(testsqlite-db) 
}


