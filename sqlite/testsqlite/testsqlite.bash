# === func-gen- : sqlite/testsqlite/testsqlite fgp sqlite/testsqlite/testsqlite.bash fgn testsqlite fgh sqlite/testsqlite
testsqlite-src(){      echo sqlite/testsqlite/testsqlite.bash ; }
testsqlite-source(){   echo ${BASH_SOURCE:-$(env-home)/$(testsqlite-src)} ; }
testsqlite-vi(){       vi $(testsqlite-source) ; }
testsqlite-env(){      elocal- ; }
testsqlite-usage(){ cat << EOU





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


testsqlite--()
{
   #cc $(testsqlite-sdir)/testsqlite.c -L/opt/local/lib -lsqlite3 -o $LOCAL_BASE/env/bin/testsqlite && testsqlite $(testsqlite-db)

   clang testsqlite.cc -L/opt/local/lib -lsqlite3 -lstdc++  -o $LOCAL_BASE/env/bin/testsqlite && DBPATH=$(testsqlite-db) testsqlite

   echo select \* from A \; | sqlite3 $(testsqlite-db) 
}

