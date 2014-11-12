ipython-src(){ echo python/ipython.bash ; }
ipython-source(){  echo $(env-home)/$(ipython-src) ; }
ipython-vi(){      vi $(ipython-source) ; }


ipython-profile-path(){ echo ~/.ipython/profile_$1/ipython_config.py ; }
ipython-edit(){ vi $(ipython-profile-path $(ipython-profile)) ;} 
ipython-profile(){ echo g4dae ; }

ipython-nb(){
    chroma-
    ipython notebook --profile $(ipython-profile)
}


ipython-usage(){ cat << EOU


IPYTHON
========

profiles
---------

::

    (chroma_env)delta:~ blyth$ ipython profile create g4dae
    [ProfileCreate] Generating default config file: u'/Users/blyth/.ipython/profile_g4dae/ipython_config.py'
    [ProfileCreate] Generating default config file: u'/Users/blyth/.ipython/profile_g4dae/ipython_notebook_config.py'
    [ProfileCreate] Generating default config file: u'/Users/blyth/.ipython/profile_g4dae/ipython_nbconvert_config.py'
    (chroma_env)delta:~ blyth$ 


plotly
-------

* https://plot.ly/python/3d-plots-tutorial/

notebooks
-----------

* http://ipython.org/ipython-doc/2/notebook/index.html

nbviewer
---------

* http://nbviewer.ipython.org/faq
* https://github.com/ipython/nbviewer


customize profile
------------------

::

    ipython-edit

::

    exec_lines = r"""
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    ph = lambda _:np.load(os.environ['DAE_PATH_TEMPLATE'] % _)
    np.set_printoptions(suppress=True, precision=3)
    """
    c.InteractiveShellApp.exec_lines = exec_lines.split("\n")





installs
----------

D
~~

Uninstall macports ipython and pip install inside chroma virtualenv, as misbehaves there otherwise::

    delta:~ blyth$ sudo port uninstall py27-ipython
    Password:
    Warning: port definitions are more than two weeks old, consider updating them by running 'port selfupdate'.
    --->  Deactivating py27-ipython @1.1.0_1+scientific
    --->  Cleaning py27-ipython
    --->  Uninstalling py27-ipython @1.1.0_1+scientific
    --->  Cleaning py27-ipython
    delta:~ blyth$ 
    delta:~ blyth$ which ipython
    delta:~ blyth$ chroma-
    (chroma_env)delta:~ blyth$ which pip
    /usr/local/env/chroma_env/bin/pip
    (chroma_env)delta:~ blyth$ pip install ipython
    Downloading/unpacking ipython
      Downloading ipython-1.2.1.tar.gz (8.7MB): 8.7MB downloaded
      Running setup.py egg_info for package ipython
        
    Installing collected packages: ipython
      Running setup.py install for ipython
        
        Installing ipcontroller script to /usr/local/env/chroma_env/bin
        Installing iptest script to /usr/local/env/chroma_env/bin
        Installing ipcluster script to /usr/local/env/chroma_env/bin
        Installing ipython script to /usr/local/env/chroma_env/bin
        Installing pycolor script to /usr/local/env/chroma_env/bin
        Installing iplogger script to /usr/local/env/chroma_env/bin
        Installing irunner script to /usr/local/env/chroma_env/bin
        Installing ipengine script to /usr/local/env/chroma_env/bin
    Successfully installed ipython
    Cleaning up...
    (chroma_env)delta:~ blyth$ 


ipython libedit issue, at ipython startup message::

    It is highly recommended that you install readline, which is easy_installable:
         easy_install readline
    Note that `pip install readline` generally DOES NOT WORK, because
    it installs to site-packages, which come *after* lib-dynload in sys.path,
    where readline is located.  It must be `easy_install readline`, or to a custom
    location on your PYTHONPATH (even --user comes after lib-dyload).

do so::

    (chroma_env)delta:~ blyth$ easy_install readline
    ...
    Running readline-6.2.4.1/setup.py -q bdist_egg --dist-dir /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/easy_install-t49Iq7/readline-6.2.4.1/egg-dist-tmp-DcEEyu
    ld: warning: ignoring file /opt/local/lib/libncurses.dylib, file was built for x86_64 which is not the architecture being linked (i386): /opt/local/lib/libncurses.dylib
    Adding readline 6.2.4.1 to easy-install.pth file

    Installed /usr/local/env/chroma_env/lib/python2.7/site-packages/readline-6.2.4.1-py2.7-macosx-10.9-x86_64.egg


refs
-------


  Good intro to pylab/numpy/ipython etc...
     http://conference.scipy.org/scipy2010/tutorials.html



   Issue with macports ipython 2.5 and readline, getting gibberish prompt
   1st try uninstall and install 

simon:qxml blyth$ sudo port uninstall py25-ipython
--->  The following versions of py25-ipython are currently installed:
--->      py25-ipython @0.9.1_0
--->      py25-ipython @0.10.2_1 (active)


    sudo port clean py25-ipython   
    sudo port install py25-ipython   -scientific

To make the Python 2.5 version of IPython the one that is run
    when you execute the commands without a version suffix, e.g. 'ipython',
        run:
	        port select --set ipython ipython25

		--->  Cleaning py25-ipython


uninstallation/installation of py25-readline + py25-ipython 
still gives a gibberized prompt...

make it less objectionable by changing config to use "colors NoColor"

   vi ~/.ipython/ipythonrc




IHEP with local python 2.5.6
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Later versions of ipython require py2.6::

	[dayabay] /home/blyth > rm -rf build
	[dayabay] /home/blyth > pip install ipython==0.10
	Downloading/unpacking ipython==0.10
	  Downloading ipython-0.10.tar.gz (5.8Mb): 5.8Mb downloaded
	  Running setup.py egg_info for package ipython
	Installing collected packages: ipython
	  Running setup.py install for ipython
	    Installing iptest script to /home/blyth/local/python/Python-2.5.6/bin
	    Installing ipythonx script to /home/blyth/local/python/Python-2.5.6/bin
	    Installing ipcluster script to /home/blyth/local/python/Python-2.5.6/bin
	    Installing ipython script to /home/blyth/local/python/Python-2.5.6/bin
	    Installing pycolor script to /home/blyth/local/python/Python-2.5.6/bin
	    Installing ipcontroller script to /home/blyth/local/python/Python-2.5.6/bin
	    Installing ipengine script to /home/blyth/local/python/Python-2.5.6/bin
	Successfully installed ipython
	Cleaning up...
	[dayabay] /home/blyth > 






EOU
}


ipython-env(){
  elocal-
}


ipython-version(){ ipython -V ; }


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
