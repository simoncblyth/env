
qmtest-usage(){
cat << EOU

vars...

QMTEST_NAME  : $QMTEST_NAME
QMTEST_HOME  : $QMTEST_HOME
QMTEST_SRC   : $QMTEST_SRC

PYTHONPATH    : $PYTHONPATH
PATH          : $PATH 
which python : $(which python)

functions...

qmtest-get      :  download and unpack
qmtest-home     :  cd \$QMTEST_HOME
qmtest-src      :  cd \$QMTEST_SRC
qmtest-install  :  install into \$QMTEST_HOME
                   

which qmtest    : $(which qmtest)


EOU
 
 
}


qmtest-learning(){
cat << EOL

 qmtest extensions  
  
 ** Available test classes ** 
- command.ExecTest Check a program's output and exit code. 
- command.ShellCommandTest Check a shell command's output and exit code. 
- command.ShellScriptTest  Check a shell script's output and exit code. 
- file.FileContentsTest   Check that the contents of a file match the expected value. 
- python.ExceptionTest    Check that the specified Python code raises an exception. 
- python.ExecTest         Check that a Python expression evaluates to true. 
- python.StringExceptionTest  Check that the specified Python code raises a string exception. 
- compilation_test.CompilationTest  A CompilationTest compiles and optionally runs an executable. 
- compilation_test.ExecutableTest  An ExecuableTest runs an executable from a CompiledResource.  
  

EOL

}




qmtest-help(){

  qmtest --help
  qmtest run --help   ## -O outcomes .... to compare to 

  ## whats the difference between outcomes and expectations 

 


}


qmtest-demo(){

  local tmp=/tmp/$FUNCNAME 
  local msg="=== $FUNCNAME :"
  
  rm -rf $tmp
  mkdir -p $tmp
  cd $tmp
  
  qmtest create-tdb
  
  echo $msg the QMTest is regarded as the \"database\"
  ls -l  QMTest/configuration   
  xmllint --format  QMTest/configuration   
  
  
  qmtest create --id=python_pass -a expression='True' test python.ExecTest 
  xmllint --format python_pass.qmt 
  
  qmtest create --id=python_fail -a expression='False' test python.ExecTest 
  xmllint --format python_fail.qmt 

  
  qmtest create --id=dir1.one -a expression='True' test python.ExecTest
  xmllint --format dir1.qms/one.qmt
  
  qmtest create --id=dir1.two -a expression='False' test python.ExecTest
  xmllint --format dir1.qms/two.qmt
  
  qmtest create --id=dir2.one -a expression='True' test python.ExecTest 
  xmllint --format dir2.qms/one.qmt
  
  qmtest create --id=dir2.dir3.one -a expression='False' test python.ExecTest
  xmllint --format dir2.qms/dir3.qms/one.qmt
  
  qmtest create --id=dir2.dir3.two -a expression='False' test python.ExecTest 
  xmllint --format dir2.qms/dir3.qms/two.qmt
  
  qmtest ls
  qmtest ls -lR
  
  qmtest run 
  ls -l results.qmr   ## not in xml 


}



qmtest-uninstall(){

## when you forgot the prefix !!

sudo rm -rf /Library/Python/2.5/site-packages/qm
sudo rm -f /Library/Python/2.5/site-packages/qmtest-2.4-py2.5.egg-info
sudo rm -rf /System/Library/Frameworks/Python.framework/Versions/2.5/share/qmtest
sudo rm -rf /System/Library/Frameworks/Python.framework/Versions/2.5/share/doc/qmtest
sudo rm -f  /usr/local/bin/qmtest

}


qmtest-env(){

  elocal-
  
  export QMTEST_NAME=qmtest-2.4
  export QMTEST_SRC=$LOCAL_BASE/qmtest/unpack/$QMTEST_NAME
  export QMTEST_HOME=$LOCAL_BASE/qmtest/$QMTEST_NAME

  export PYTHONPATH=$QMTEST_HOME/lib/python2.5/site-packages
  export PATH=$QMTEST_HOME/bin:$PATH  
  
}




qmtest-home(){ cd $QMTEST_HOME ; }
qmtest-src(){ cd $QMTEST_SRC ; }


qmtest-get(){

  local nik=qmtest 
  local name=$QMTEST_NAME
  local tgz=$name.tar.gz
  
  local url=http://www.codesourcery.com/public/$nik/$name/$tgz
  
  cd $LOCAL_BASE
  [ ! -d $nik ] && $SUDO mkdir $nik && $SUDO chown $USER $nik

  local dir=$nik/unpack  
  mkdir -p $dir
  
  cd $dir
  
  [ ! -f $tgz  ] && curl -O $url
  [ ! -d $name ] && tar zxvf $tgz
  

}

qmtest-install(){

   local msg="=== $FUNCNAME :"
   
   qmtest-src
   
   echo $msg standard build and install into QMTEST_HOME $QMTEST_HOME ... answer YES to continue
   read answer
   [ "$answer" != "YES" ] && return 1
   
   local home=$QMTEST_HOME
   mkdir -p $home
   
   python setup.py install --prefix=$home
   	 
}