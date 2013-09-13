# === func-gen- : sqlite/sqlite3 fgp sqlite/sqlite3.bash fgn sqlite3 fgh sqlite
sqlite3-src(){      echo sqlite/sqlite3.bash ; }
sqlite3-source(){   echo ${BASH_SOURCE:-$(env-home)/$(sqlite3-src)} ; }
sqlite3-vi(){       vi $(sqlite3-source) ; }
sqlite3-env(){      elocal- ; }
sqlite3-usage(){ cat << EOU

SQLITE3
========

* http://www.sqlite.org/download.html

yum depends on sqlite3


EOU
}
sqlite3-dir(){ echo $(local-base)/env/sqlite/$(sqlite3-name) ; }
sqlite3-prefix(){ echo $(local-base)/env ; }

sqlite3-cd(){  cd $(sqlite3-dir); }
sqlite3-mate(){ mate $(sqlite3-dir) ; }
sqlite3-name(){ echo sqlite-autoconf-3080002 ; }
sqlite3-url(){  echo http://www.sqlite.org/2013/$(sqlite3-name).tar.gz ; }
sqlite3-get(){
   local dir=$(dirname $(sqlite3-dir)) &&  mkdir -p $dir && cd $dir

   local url=$(sqlite3-url)
   local tgz=$(basename $url)
   local nam=${tgz/.tar.gz}

   [ ! -f "$tgz" ] && curl -L -O "$url"
   [ ! -d "$nam" ] && tar zxvf $tgz

}

sqlite3-path(){ PATH=$(sqlite3-prefix)/bin:$PATH ; }
sqlite3--(){ $(sqlite3-prefix)/bin/sqlite3 $* ; }

sqlite3-build(){
  sqlite3-configure
  sqlite3-make
  sqlite3-install
}


sqlite3-configure(){
   sqlite3-cd
   ./configure --prefix=$(sqlite3-prefix) 
}

sqlite3-make(){
   sqlite3-cd
   make 
}

sqlite3-install(){
   sqlite3-cd
   sudo make install
}





