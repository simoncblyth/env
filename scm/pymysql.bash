
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

   # http://initd.org/pub/software/pysqlite/doc/install-source.html
   # NB the default python in path is too old: 2.2

   nam=$PYSQLITE_NAME
   cd $LOCAL_BASE/pysqlite/build/$nam

   perl -pi -e "s|^(include_dirs=)(.*)$|\$1$SQLITE_HOME/include|g" setup.cfg
   perl -pi -e "s|^(library_dirs=)(.*)$|\$1$SQLITE_HOME/lib|g"     setup.cfg

  # doesnt work with this version of python 
  #   /usr/bin/python2.4 setup.py build

   $PYTHON_HOME/bin/python setup.py build
   $PYTHON_HOME/bin/python setup.py install 

   # installs as a site-package into the installed python at
   #      $PYTHON_HOME/lib/python2.5/site-packages/pysqlite2
   #
}

pymysql-test(){

    # need to run the test from a directory other than the build directory
    #  otherwise it fails with
    #   ImportError: No module named _sqlite
    cd 
    $PYTHON_HOME/bin/python << EOP
from pysqlite2 import test
test.test()
EOP

 
}
