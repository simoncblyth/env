

pexpect-get(){

   
  nam=pexpect-2.1
  nik=pexpect
  tgz=$nam.tar.gz

  url=http://jaist.dl.sourceforge.net/sourceforge/pexpect/$tgz

  cd $LOCAL_BASE
  test -d $nik || ( $SUDO mkdir $nik && $SUDO chown $USER $nik )
  cd $nik

  test -f $tgz || curl -o $tgz $url
  test -d $nam || tar -zxvf $tgz 

  
  cd $nam


  
  # python setup.py install 
  # Writing /usr/local/python/Python-2.5.1/lib/python2.5/site-packages/pexpect-2.1-py2.5.egg-info 

  # sudo /usr/local/bin/python setup.py install
  # Writing /Library/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/pexpect-2.1-py2.5.egg-info

}



