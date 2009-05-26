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


   Django says : MySQLdb-1.2.1p2 or newer is required


EOU

}


yum-installed-(){ yum list installed | grep $1 ; }

pymysql-preq-(){ echo mysql-server mysql-devel ; }
pymysql-preq(){
  local msg="=== $FUNCNAME :"
  local preqs=$(pymysql-preq-)
  echo $msg $preqs
  local preq
  for preq in $preqs ; do 
    ! yum-installed- $preq  && sudo yum install $preq
  done
}

pymysql-build(){
   local msg="=== $FUNCNAME :"
   pymysql-preq
   pymysql-get
   pymysql-setup build

   pymysql-install 
   pymysql-test
}

pymysql-test(){ python -c "import MySQLdb " ; }
pymysql-name(){ echo MySQL-python-1.2.2 ; }
pymysql-fold(){ echo $(local-system-base)/pymysql ; }
pymysql-dir(){  echo $(pymysql-fold)/$(pymysql-name) ; }
pymysql-cd(){   cd $(pymysql-dir) ; }

pymysql-get(){
  
  local nam=$(pymysql-name)
  local tgz=$nam.tar.gz
  local url=http://jaist.dl.sourceforge.net/sourceforge/mysql-python/$tgz
 
  local iwd=$PWD
  local dir=$(pymysql-fold)
  [ ! -d "$dir" ] && mkdir -p "$dir"
  cd $dir
  [ ! -f $tgz ] && curl -L -O $url
  [ ! -d $nam ] && tar  -zxvf $tgz 
  cd $iwd
}

pymysql-setup(){
   local msg="=== $FUNCNAME :"
   local iwd=$PWD
   pymysql-cd
   [ "$(which mysql_config)" == "" ] && echo $msg ABORT no mysql_config in path && return 1
   python setup.py $*
   cd $iwd
}

pymysql-install(){
   local msg="=== $FUNCNAME :"
   local iwd=$PWD
   pymysql-cd
   local cmd="sudo python setup.py install "
   echo $msg $cmd
   eval $cmd
   python setup.py build
   cd $iwd
}


