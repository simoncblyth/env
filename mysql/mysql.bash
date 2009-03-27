
mysql-env(){    elocal- ; }
mysql-src(){ echo mysql/mysql.bash ; }
mysql-source(){ echo ${BASH_SOURCE:-$(env-home)/$(mysql-src)} ; }
mysql-vi(){     vi $(mysql-source) ; }
mysql-usage(){
  cat << EOU

    Installed 4.1.22 with yum $(env-wikiurl)/MySQL onto C2

        http://dev.mysql.com/doc/refman/4.1/en/


EOU


}

mysql-install(){
   sudo yum install mysql-server
}


