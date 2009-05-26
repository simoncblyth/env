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

pymysql-test(){
   python -c "import MySQLdb "
}

pymysql-cd(){ cd $(pymysql-dir) ; }

pymysql-name(){ echo MySQL-python-1.2.2 ; }
pymysql-dir(){ echo $(local-system-base)/pymysql ; }
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
  [ ! -d build ] && mkdir build
  [ ! -d build/$nam ] && tar -C build -zxvf $tgz 
  cd $iwd
}

pymysql-install(){
  
   local nam=$(pymysql-name)
   local nik=pymysql
   local iwd=$PWD
   cd $(pymysql-dir)/build/$nam
 
}

pymysql-test(){
    python << EOP
from pysqlite2 import test
test.test()
EOP
}
