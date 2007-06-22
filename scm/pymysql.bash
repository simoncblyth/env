
#
# http://www.sqlalchemy.org/
#
#
#
#


PYMYSQL_NAME=MySQL-python-1.2.2

pymysql-get(){
  
  nam=$PYMYSQL_NAME
  tgz=$nam.tar.gz
  url=http://jaist.dl.sourceforge.net/sourceforge/mysql-python/$tgz
  nik=pymysql
  
  cd $LOCAL_BASE
  test -d $nik || ( $SUDO mkdir $nik && $SUDO chown $USER $nik )
  cd $nik

  test -f $tgz || curl -o $tgz $url
  test -d build || mkdir build
  test -d build/$nam || tar -C build -zxvf $tgz 
}

pymysql-install(){
  
   local nam=$PYMYSQL_NAME
   local nik=pymysql
   cd $LOCAL_BASE/$nik/build/$nam

   local ipath=$PATH
   
   #
   # need MAMPs mysql_config to be in the path 
   # /Applications/MAMP/Library/bin
   # NB will need to rebuild on updating MAMP ... imminent 
   #
   
   export PATH=/Applications/MAMP/Library/bin:$PATH
   $PYTHON_HOME/bin/python setup.py build
   #sudo $PYTHON_HOME/bin/python setup.py install 
   export PATH=$ipath  
#
# running build_ext
# building '_mysql' extension
# creating build/temp.macosx-10.4-ppc-2.5
# gcc -fno-strict-aliasing -Wno-long-double -no-cpp-precomp -mno-fused-madd -DNDEBUG -g -O3 -Wall -Wstrict-prototypes -Dversion_info=(1,2,2,'final',0) -D__version__=1.2.2 -I/Applications/MAMP/Library/include/mysql -I/usr/local/python/Python-2.5.1/include/python2.5 -c _mysql.c -o build/temp.macosx-10.4-ppc-2.5/_mysql.o -fno-omit-frame-pointer
# _mysql.c:35:23: error: my_config.h: No such file or directory
# _mysql.c:40:19: error: mysql.h: No such file or directory
# _mysql.c:41:26: error: mysqld_error.h: No such file or directory
# _mysql.c:42:20: error: errmsg.h: No such file or directory
# _mysql.c:78: error: parse error before 'MYSQL'
#
#
#  MAMP does not come with headers !!!
# [g4pb:/usr/local/pymysql/build/MySQL-python-1.2.2] blyth$ find /Applications/newMAMP/ -name '*.h'
# [g4pb:/usr/local/pymysql/build/MySQL-python-1.2.2] blyth$ find /Applications/MAMP/ -name '*.h'
#

}

pymysql-test(){

       cd 
    $PYTHON_HOME/bin/python << EOP
from pysqlite2 import test
test.test()
EOP

 
}
