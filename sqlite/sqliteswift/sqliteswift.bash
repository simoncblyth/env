# === func-gen- : sqlite/sqliteswift/sqliteswift fgp sqlite/sqliteswift/sqliteswift.bash fgn sqliteswift fgh sqlite/sqliteswift
sqliteswift-src(){      echo sqlite/sqliteswift/sqliteswift.bash ; }
sqliteswift-source(){   echo ${BASH_SOURCE:-$(env-home)/$(sqliteswift-src)} ; }
sqliteswift-vi(){       vi $(sqliteswift-source) ; }
sqliteswift-env(){      elocal- ; }
sqliteswift-usage(){ cat << EOU

SQLite Swift Wrapper
======================

A type-safe, Swift-language layer over SQLite3.

* https://github.com/stephencelis/SQLite.swift/blob/master/Documentation/Index.md




EOU
}

sqliteswift-name(){ echo SQLite.swift ; }
sqliteswift-dir(){ echo $(local-base)/env/sqlite/$(sqliteswift-name) ; }
sqliteswift-cd(){  cd $(sqliteswift-dir); }
sqliteswift-mate(){ mate $(sqliteswift-dir) ; }
sqliteswift-get(){
   local dir=$(dirname $(sqliteswift-dir)) &&  mkdir -p $dir && cd $dir

   [ ! -d $(sqliteswift-name) ] && git clone https://github.com/stephencelis/SQLite.swift.git

}
