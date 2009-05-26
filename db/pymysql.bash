pymysql-src(){ echo db/pymysql.bash ; }
pymysql-source(){ echo ${BASH_SOURCE:-$(env-home)/$(pymysql-src)} ; }
pymysql-vi(){     vi $(pymysql-source) ; }
pymysql-env(){
   local-
   python-
   export PYMYSQL_NAME=$(pymysql-name)
}
pymqsql-usage(){
   cat << EOU

   If your system python + MySQL are new enough for what you want to 
   do you should use your system MySQL-python rather than use this
   more involved source approach.  


      pymysql-name : $(pymysql-name)

EOU

}


pymysql-build(){
   easy_install MySQL-python 
}

pymysql-test(){ python -c "import MySQLdb " ;  }
pymysql-name(){ echo MySQL-python-1.2.2 ; }
pymysql-fold(){ echo $(local-system-base)/pymysql ; }
pymysql-dir(){  echo $(pymysql-fold)/$(pymysql-name) ; }
pymysql-cd(){   cd $(pymysql-dir) ; }
pymysql-get(){
  
  local nam=$(pymysql-name)
  local tgz=$nam.tar.gz
  local url=http://jaist.dl.sourceforge.net/sourceforge/mysql-python/$tgz
  local nik=pymysql
 
  local iwd=$PWD
  local dir=$(pymysql-dir)
  [ ! -d "$dir" ] && mkdir -p "$dir"
  cd $dir
  [ ! -f $tgz ] && curl -L -O $url
  [ ! -d $nam ] && tar -zxvf $tgz 
  cd $iwd
}

pymysql-build(){
   local msg="=== $FUNCNAME :"
   local iwd=$PWD
   pymysql-cd
   [ "$(which mysql_config)" == "" ] && echo $msg ABORT need mysql_config in path && return 1
   python setup.py build
   cd $iwd
}

pymysql-install(){
   local msg="=== $FUNCNAME :"
   local iwd=$PWD
   pymysql-cd
   python setup.py install
   cd $iwd
}
