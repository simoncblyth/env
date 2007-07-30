
#  prerequisites to trac :
#
#      svn
#      apache2
#      sqlite
#      pysqlite
#      python
#      swig
#      modpython OR modwsgi
#      clearsilver
#
#
#
#
#   trac-site-wipe
#   trac-wipe
#   trac-get
#   trac-install
#   


trac-site-wipe(){

  cd $PYTHON_SITE && sudo rm -rf trac && sudo rm -rf trac*.egg-info && ls -alst $PYTHON_SITE 
}

trac-wipe(){

   nam=$TRAC_NAME
   nik=$TRAC_NIK
   cd $LOCAL_BASE/$nik
   rm -rf build/$nam
}



trac-get(){
  
  nam=$TRAC_NAME
  tgz=$nam.tar.gz
  url=http://ftp.edgewall.com/pub/trac/$tgz

  cd $LOCAL_BASE
  test -d trac || ( $SUDO mkdir trac && $SUDO chown $USER trac )
  cd trac  

  test -f $tgz || curl -o $tgz $url
  test -d build || mkdir build
  test -d build/$nam || tar -C build -zxvf $tgz 
}

trac-install(){

  nam=$TRAC_NAME
  cd $LOCAL_BASE/trac/build/$nam

  $PYTHON_HOME/bin/python ./setup.py --help    ## not very helpful
  $PYTHON_HOME/bin/python ./setup.py install --help   ## more helpful
  $PYTHON_HOME/bin/python ./setup.py install 

  #
  # creates and installs into :
  #
  #      $PYTHON_HOME/lib/python2.5/site-packages/trac
  #      $PYTHON_HOME/share/trac/templates
  #      $PYTHON_HOME/share/trac/htdocs
  #
  #      $PYTHON_HOME/bin/trac-admin
  #      $PYTHON_HOME/bin/tracd
  #
}

