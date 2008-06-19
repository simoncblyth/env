

ipython-env(){
  elocal-
}


ipython-get(){


  local msg="=== $FUNCNAME "

  local nik=ipython
  #local nam=$nik-0.8.1
  local nam=$nik-0.8.2
  local tgz=$nam.tar.gz
  local url=http://ipython.scipy.org/dist/$tgz

  local dir=$LOCAL_BASE/python/$nik   
   
   echo $msg 
   
  mkdir -p $dir || return 1
  cd $dir
  
  test -f $tgz || curl -o $tgz $url
  test -d $nam || tar zxvf $tgz
 
  cd $nam
  
 # 
 # unix python OR MacPython 
 #
 # local py=python
 #  local py="sudo /usr/local/bin/python"
 
  echo $msg installing into the python in your path $(which python) ===
  which python 
 
   python -c "import sys;print sys.prefix"
   $SUDO python setup.py install
 
 #
 #  this simple python switch , isnt working tis sensitive to the environment ... so 
 #
 
 
  
 
 
}

ipython-rm(){

  local dir=$(dirname $(which ipython))
  

}



ipython-check(){

    local p=$(which python)
    local i=$(which ipython)
    
    [ "$(dirname $p)" == "$(dirname $i)" ] && echo "1" || echo "0" 
}

ipython-fix(){

   local chk=$(ipython-check)
   if [ $chk == 1 ]; then
       local i=$(which ipython)
       echo === ipython-fix editing the ipython script $i 
       perl -i.orig -pe '$. == 1 && s/#!.*/#!\/usr\/bin\/env python/; '  $i
       diff $i.orig $i
    else
       echo === ipython-fix paths to python and ipython must be from the same folder 
    fi



}



ipython-readline(){

   # recipe from http://ipython.scipy.org/moin/InstallationOSXLeopard 
   # BUT turns out that the egg is in pypi already so can use the ez solution
   #     http://pypi.python.org/pypi/readline/2.5.1
   #

   local iwd=$PWD
   local dir=$LOCAL_BASE/env/ipython && mkdir -p $dir
    cd $dir
   
   local nam=python-readline-leopard
   local tgz=$nam-011808.tar.gz
   local url=http://ipython.scipy.org/moin/InstallationOSXLeopard?action=AttachFile\&do=get\&target=$tgz
  
   [ ! -f $tgz ] && curl -o $tgz $url
   [ ! -d $nam ] && tar zxvf $tgz
   
   cd $nam
   tgz=readline-5.2.tar.gz
   url=http://ftp.gnu.org/gnu/readline/$tgz
   
   egg=readline-2.5.1-py2.5-macosx-10.5-fat.egg
   
   [ ! -f $tgz ] && curl -o $tgz $url
   [ ! -f $egg ] && ./build.sh

   #cd $iwd
}

ipython-readline-ez(){
   easy_install readline==2.5.1
}


ipython-easyinstall-log(){

   # hmm seem to have double egged the pudding ?

cat << EOL

 easy_install readline-2.5.1-py2.5-macosx-10.5-fat.egg 
Processing readline-2.5.1-py2.5-macosx-10.5-fat.egg
creating /usr/local/dyb/trunk_dbg/external/Python/2.5/osx105_ppc_gcc401/lib/python2.5/site-packages/readline-2.5.1-py2.5-macosx-10.5-fat.egg
Extracting readline-2.5.1-py2.5-macosx-10.5-fat.egg to /usr/local/dyb/trunk_dbg/external/Python/2.5/osx105_ppc_gcc401/lib/python2.5/site-packages
Adding readline 2.5.1 to easy-install.pth file

Installed /usr/local/dyb/trunk_dbg/external/Python/2.5/osx105_ppc_gcc401/lib/python2.5/site-packages/readline-2.5.1-py2.5-macosx-10.5-fat.egg
Processing dependencies for readline==2.5.1
Searching for readline==2.5.1
Reading http://pypi.python.org/simple/readline/
Reading http://www.python.org/
Best match: readline 2.5.1
Downloading http://pypi.python.org/packages/2.5/r/readline/readline-2.5.1-py2.5-macosx-10.5-ppc.egg#md5=25ebe33023a003c8bb8ba7507944f29c
Processing readline-2.5.1-py2.5-macosx-10.5-ppc.egg
creating /usr/local/dyb/trunk_dbg/external/Python/2.5/osx105_ppc_gcc401/lib/python2.5/site-packages/readline-2.5.1-py2.5-macosx-10.5-ppc.egg
Extracting readline-2.5.1-py2.5-macosx-10.5-ppc.egg to /usr/local/dyb/trunk_dbg/external/Python/2.5/osx105_ppc_gcc401/lib/python2.5/site-packages
Removing readline 2.5.1 from easy-install.pth file
Adding readline 2.5.1 to easy-install.pth file

Installed /usr/local/dyb/trunk_dbg/external/Python/2.5/osx105_ppc_gcc401/lib/python2.5/site-packages/readline-2.5.1-py2.5-macosx-10.5-ppc.egg
Finished processing dependencies for readline==2.5.1

EOL

}





# WARNING: Readline services not available on this platform.
# WARNING: The auto-indent feature requires the readline library
# WARNING: Proper color support under MS Windows requires the pyreadline library.
# You can find it at:
# http://ipython.scipy.org/moin/PyReadline/Intro
# Gary's readline needs the ctypes module, from:
# http://starship.python.net/crew/theller/ctypes
# (Note that ctypes is already part of Python versions 2.5 and newer).
#
# Defaulting color scheme to 'NoColor'
# Python 2.5.1 (r251:54863, May  8 2007, 22:27:26) 
# Type "copyright", "credits" or "license" for more information.
# 
# IPython 0.8.1 -- An enhanced Interactive Python.
# ?       -> Introduction to IPython's features.
# %magic  -> Information about IPython's 'magic' % functions.
# help    -> Python's own help system.
# object? -> Details about 'object'. ?object also works, ?? prints more.
#