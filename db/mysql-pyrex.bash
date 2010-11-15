# === func-gen- : db/mysql-pyrex fgp db/mysql-pyrex.bash fgn mysql-pyrex fgh db
mysql-pyrex-src(){      echo db/mysql-pyrex.bash ; }
mysql-pyrex-source(){   echo ${BASH_SOURCE:-$(env-home)/$(mysql-pyrex-src)} ; }
mysql-pyrex-vi(){       vi $(mysql-pyrex-source) ; }
mysql-pyrex-env(){      elocal- ; }
mysql-pyrex-usage(){
  cat << EOU
     mysql-pyrex-src : $(mysql-pyrex-src)
     mysql-pyrex-dir : $(mysql-pyrex-dir)


EOU
}
mysql-pyrex-name(){ echo mysql-pyrex-0.9.0a1 ; }
mysql-pyrex-dir(){ echo $(local-base)/env/db/$(mysql-pyrex-name) ; }
mysql-pyrex-cd(){  cd $(mysql-pyrex-dir); }
mysql-pyrex-mate(){ mate $(mysql-pyrex-dir) ; }
mysql-pyrex-get(){
   local dir=$(dirname $(mysql-pyrex-dir)) &&  mkdir -p $dir && cd $dir
   local nam=$(mysql-pyrex-name)
   local tgz=$nam.tar.gz
   [ ! -f "$tgz" ] && curl -O http://ehuss.org/mysql/download/$tgz
   [ ! -d "$nam" ] && tar zxvf $tgz
}
