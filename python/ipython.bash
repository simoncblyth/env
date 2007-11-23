
ipython-get(){

  local nik=ipython
  local nam=$nik-0.8.1
  local tgz=$nam.tar.gz
  local url=http://ipython.scipy.org/dist/$tgz

  local dir=$LOCAL_BASE/python/$nik   
   
  mkdir -p $dir 
  cd $dir
  
  test -f $tgz || curl -o $tgz $url
  test -d $nam || tar zxvf $tgz
 
  cd $nam
  
 # 
 # unix python OR MacPython 
 #
 # local py=python
 #  local py="sudo /usr/local/bin/python"
 
  echo === ipython-get installing into the python in your path $(which python) ===
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