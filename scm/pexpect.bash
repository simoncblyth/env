
pexpect-env(){

  local msg="=== $FUNCNAME :"
}


pexpect-get(){

   
  local nam=pexpect-2.1
  local nik=pexpect
  local tgz=$nam.tar.gz

  local url=http://jaist.dl.sourceforge.net/sourceforge/pexpect/$tgz

  cd $LOCAL_BASE
  test -d $nik || ( $SUDO mkdir $nik && $SUDO chown $USER $nik )
  cd $nik

  test -f $tgz || curl -o $tgz $url
  test -d $nam || tar -zxvf $tgz 

  
  cd $nam


  

}

pexpect-install(){


  python setup.py install
  # on hfag
  # Writing /data/usr/local/python/Python-2.5.1/lib/python2.5/site-packages/pexpect-2.1-py2.5.egg-info 

  # python setup.py install 
  # Writing /usr/local/python/Python-2.5.1/lib/python2.5/site-packages/pexpect-2.1-py2.5.egg-info 

  # sudo /usr/local/bin/python setup.py install
  # Writing /Library/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/pexpect-2.1-py2.5.egg-info

  # NB on grid1 installed as dayabaysoft user ... tag P,  the same python as used when blyth at tag G1
  # [dayabaysoft@grid1 pexpect-2.1]$ which python
  # Writing /disk/d4/dayabay/local/python/Python-2.5.1/lib/python2.5/site-packages/pexpect-2.1-py2.5.egg-info





}

