# === func-gen- : db/mysqludfsys fgp db/mysqludfsys.bash fgn mysqludfsys fgh db
mysqludfsys-src(){      echo db/mysqludfsys.bash ; }
mysqludfsys-source(){   echo ${BASH_SOURCE:-$(env-home)/$(mysqludfsys-src)} ; }
mysqludfsys-vi(){       vi $(mysqludfsys-source) ; }
mysqludfsys-env(){      elocal- ; }
mysqludfsys-usage(){ cat << EOU


  http://www.mysqludf.org/lib_mysqludf_sys/index.php

  http://dev.mysql.com/doc/refman/5.1/en/udf-compiling.html

EOU
}

mysqludfsys-nam(){  echo lib_mysqludf_sys ; }
mysqludfsys-name(){ echo lib_mysqludf_sys_0.0.3 ; }
mysqludfsys-dir(){ echo $(local-base)/env/db/$(mysqludfsys-name) ; }
mysqludfsys-cd(){  cd $(mysqludfsys-dir); }
mysqludfsys-mate(){ mate $(mysqludfsys-dir) ; }
mysqludfsys-get(){
   local dir=$(dirname $(mysqludfsys-dir)) &&  mkdir -p $dir && cd $dir
   local nam=$(mysqludfsys-name)
   local url=http://www.mysqludf.org/lib_mysqludf_sys/$nam.tar.gz
   local tgz=$(basename $url)
   [ ! -d "$nam" ] && mkdir -p $nam && curl $url -o $nam/$nam.tar.gz && tar zxvf $nam/$nam.tar.gz -C $nam

}

mysqludfsys-make(){

   local nam=$(mysqludfsys-nam)
   gcc -bundle -o $nam.so $nam.c -I/opt/local/include/mysql5/mysql
   #  sudo cp lib_mysqludf_sys.so /usr/lib/

}


