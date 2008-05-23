
nose-usage(){
cat << EOU

  http://www.somethingaboutorange.com/mrl/projects/nose/
  
  \$NOSE_NAME   : $NOSE_NAME
  
  nose-get     : dl and unpack
  nose-install : build and install into \$NOSE_HOME : $NOSE_HOME
  nose-home    : cd \$NOSE_HOME
  
  \$(which nosetests)   : $(which nosetests)
  

EOU

}



nose-plugin-get(){

  cd $NOSE_HOME
  local tmp=/tmp/env/$FUNCNAME && mkdir -p $tmp
  cd $tmp
  local name=html_plugin
  local home=$NOSE_HOME
  
  
  [ ! -d $name ] && svn export http://python-nose.googlecode.com/svn/trunk/examples/$name

  cd $name
  
  python setup.py install --prefix=$home

#
# after that   "nosetests --help "  says ...
#  
#   --with-html-output    Enable plugin HtmlOutput: Output test results as ugly,
#                         unstyled html.  [NOSE_WITH_HTML_OUTPUT]

}


nose-plugin-test(){

   nosetests --with-html-output $ENV_HOME/unittest/romantest.py 2> out.html
 
   
 
}




nose-env(){

  elocal-
  
  #export NOSE_NAME=nose-0.9.3
  export NOSE_NAME=nose-0.10.2
  export NOSE_HOME=$LOCAL_BASE/nose/$NOSE_NAME

  export PYTHONPATH=$NOSE_HOME/lib/python2.5/site-packages:$PYTHONPATH
  export PATH=$NOSE_HOME/bin:$PATH  

}

nose-src(){
  cd $LOCAL_BASE/nose/unpack/$NOSE_NAME 
}

nose-home(){
  cd $NOSE_HOME
}

nose-get(){

  local nik=nose 
  local nam=$NOSE_NAME 
  local tgz=$nam.tar.gz
  local url=http://www.somethingaboutorange.com/mrl/projects/nose/$tgz
  
  
  cd $LOCAL_BASE
  
  [ ! -d $nik ] && $SUDO mkdir $nik && $SUDO chown $USER $nik

  local dir=$nik/unpack  
  mkdir -p $dir
  
  cd $dir
  
  [ ! -f $tgz  ] && curl -O $url
  [ ! -d $nam  ] && tar zxvf $tgz

}

nose-install(){

  nose-src
  local home=$NOSE_HOME
  local site=$home/lib/python2.5/site-packages
  
  #mkdir -p $home
  mkdir -p $site
  #python setup.py install --prefix=$home
  
  PYTHONPATH=$site python setup.py install --prefix=$home
  

}


nose-install-fail1(){

cat << EOF

Checking .pth file support in /usr/local/nose/nose-0.9.3/lib/python2.5/site-packages/
error: can't create or remove files in install directory

The following error occurred while trying to add or remove files in the
installation directory:

    [Errno 2] No such file or directory: '/usr/local/nose/nose-0.9.3/lib/python2.5/site-packages/test-easy-install-33466.pth'

The installation directory you specified (via --install-dir, --prefix, or
the distutils default setting) was:

    /usr/local/nose/nose-0.9.3/lib/python2.5/site-packages/

This directory does not currently exist.  Please create it and try again, or
choose a different installation directory (using the -d or --install-dir
option).

EOF

}

nose-install-fail2(){

cat << EOF

Checking .pth file support in /usr/local/nose/nose-0.9.3/lib/python2.5/site-packages/
/System/Library/Frameworks/Python.framework/Versions/2.5/Resources/Python.app/Contents/MacOS/Python -E -c pass
TEST FAILED: /usr/local/nose/nose-0.9.3/lib/python2.5/site-packages/ does NOT support .pth files
error: bad install directory or PYTHONPATH

You are attempting to install a package to a directory that is not
on PYTHONPATH and which Python does not read ".pth" files from.  The
installation directory you specified (via --install-dir, --prefix, or
the distutils default setting) was:

    /usr/local/nose/nose-0.9.3/lib/python2.5/site-packages/

and your PYTHONPATH environment variable currently contains:

    ''

Here are some of your options for correcting the problem:

* You can choose a different installation directory, i.e., one that is
  on PYTHONPATH or supports .pth files

* You can add the installation directory to the PYTHONPATH environment
  variable.  (It must then also be on PYTHONPATH whenever you run
  Python and want to use the package(s) you are installing.)

* You can set up the installation directory to support ".pth" files by
  using one of the approaches described here:

  http://peak.telecommunity.com/EasyInstall.html#custom-installation-locations

Please make the appropriate changes for your system and try again.

EOF







}




