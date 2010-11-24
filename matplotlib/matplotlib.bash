# === func-gen- : matplotlib/matplotlib fgp matplotlib/matplotlib.bash fgn matplotlib fgh matplotlib
matplotlib-src(){      echo matplotlib/matplotlib.bash ; }
matplotlib-source(){   echo ${BASH_SOURCE:-$(env-home)/$(matplotlib-src)} ; }
matplotlib-vi(){       vi $(matplotlib-source) ; }
matplotlib-env(){      elocal- ; }
matplotlib-usage(){
  cat << EOU
     matplotlib-src : $(matplotlib-src)
     matplotlib-dir : $(matplotlib-dir)



   Dependencies ...
     || numpy >= 1.1 ||  pip install numpy  ||  1.5.0 on C  ||


     http://matplotlib.sourceforge.net/faq/installing_faq.html#install-svn

         using "python setupegg.py develop" means 
             * py only updates with -update 
             * C/C++ updates require another -build


   = build problem with libpng  =

   Warning at start :
{{{
      libpng: found, but unknown version (no pkg-config)
           * Could not find 'libpng' headers in any of
           * '/usr/local/include', '/usr/include', '.'
}}}

   And build fails :
{{{
gcc -pthread -fno-strict-aliasing -DNDEBUG -g -O3 -Wall -Wstrict-prototypes -fPIC -DPY_ARRAY_UNIQUE_SYMBOL=MPL_ARRAY_API -DPYCXX_ISO_CPP_LIB=1 -I/usr/local/include -I/usr/include -I. -I/data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/numpy/core/include -I. -I/data/env/system/python/Python-2.5.1/include/python2.5 -c src/_png.cpp -o build/temp.linux-i686-2.5/src/_png.o
cc1plus: warning: command line option "-Wstrict-prototypes" is valid for Ada/C/ObjC but not for C++
src/_png.cpp:10:20: png.h: No such file or directory
src/_png.cpp:57: error: variable or field `write_png_data' declared void      
}}}

   After  {{{sudo yum install libpng-devel}}} pkg-config recognizes '''libpng'''


   Clean by removing ignored directories and files :
{{{
     svn st --no-ignore
     svn st --no-ignore | perl -p -e 's,^[I?],rm -rf ,' - | sh     ## CAUTION CHECK THE DELETION LIST BEFORE DOING THIS 
}}}


  = trying matplotlib from ipython ...  cannot "plt.show()"  =

     ipython -pylab 
{{{
/data/env/local/env/matplotlib/matplotlib/lib/matplotlib/backends/__init__.py:41: UserWarning: 
Your currently selected backend, 'agg' does not support show().
Please select a GUI backend in your matplotlibrc file ('/data/env/local/env/matplotlib/matplotlib/lib/matplotlib/mpl-data/matplotlibrc')
or with matplotlib.use()
  (backend, matplotlib.matplotlib_fname()))
}}}


{{{
...
OPTIONAL BACKEND DEPENDENCIES
                libpng: 1.2.7
               Tkinter: no
                        * TKAgg requires Tkinter
                  Gtk+: no
                        * Building for Gtk+ requires pygtk; you must be able
                        * to "import gtk" in your build/install environment
}}}


 == missing deps :  tcl-devel tk-devel tkinter ==

=== On C (a source python node) : ===

    sudo yum install tcl-devel
    sudo yum install tk-devel

 After which rebuild python (just a few min partial build):

    pythonbuild-
    pythonbuild-configure
    pythonbuild-install
    python -c "import Tkinter"

=== On N (system python node) : ===

    sudo yum install tkinter 
    python -c "import Tkinter"
    
     sudo yum install tk-devel      
     sudo yum install pygtk2-devel 


 Propagate to matplotlib by cleaning and rebuilding as described above

   Test it worked ...
        ipython -pylab
        > figure()      ## should popup a GUI window 

 == matplotlib 0.91.1 not compatible with numpy 2.0? ==


Try {{{ipython -pylab}}} :
{{{
  File "/data1/env/local/env/v/npy/lib/python2.4/site-packages/matplotlib/numerix/ma/__init__.py", line 16, in ?
    from numpy.core.ma import *
ImportError: No module named ma
}}}


 Tried {{{pip -v install matplotlib==dev}}} but no found on pypi etc.., so use the source install technique : 

{{{
(npy)[blyth@belle7 mysql_np]$ pip -v install -e svn+https://matplotlib.svn.sourceforge.net/svnroot/matplotlib/trunk/matplotlib/#egg=matplotlib
Obtaining matplotlib from svn+https://matplotlib.svn.sourceforge.net/svnroot/matplotlib/trunk/matplotlib/#egg=matplotlib
  Checking out https://matplotlib.svn.sourceforge.net/svnroot/matplotlib/trunk/matplotlib/ to /data1/env/local/env/v/npy/src/matplotlib
  Found command 'svn' at '/usr/bin/svn'
}}}
 

 == matplotlib 1.0??? installed by pip ... no plots showing on C ==

   * remember that are using source python  C ...
                  try : pip install PyGTK
     but fails with ... ImportError: No module named dsextras


     on N comes via {{{yum info pygtk2}}}


   After cleaning try a standard ... (not 1.00 and not using virtual but into source python)
       [blyth@cms01 ~]$ pip install matplotlib

   
matplotlib-test
$HOME=/home/blyth
CONFIGDIR=/home/blyth/.matplotlib
matplotlib data path /data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/matplotlib/mpl-data
loaded rc file /data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/matplotlib/mpl-data/matplotlibrc
matplotlib version 0.91.1
verbose.level helpful
interactive is False
units is False
platform is linux2
numerix numpy 1.5.0
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/pylab.py", line 1, in <module>
    from matplotlib.pylab import *
  File "/data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/matplotlib/pylab.py", line 206, in <module>
    from matplotlib.numerix import npyma as ma
  File "/data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/matplotlib/numerix/__init__.py", line 166, in <module>
    __import__('ma', g, l)
  File "/data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/matplotlib/numerix/ma/__init__.py", line 16, in <module>
    from numpy.core.ma import *
ImportError: No module named ma
[blyth@cms01 ~]$ 
[blyth@cms01 ~]$ 
[blyth@cms01 ~]$ python -c "import numpy as np ; print np.__version__ "
1.5.0

         this was the reason i moved to 1.0 from source  



    so ...
           matplotlib-get
           matplotlib-build


EOU
}
matplotlib-dir(){ 
  [ -n "$VIRTUAL_ENV" ] && echo $VIRTUAL_ENV/src/matplotlib ||  echo $(local-base)/env/matplotlib/matplotlib ; 
}
matplotlib-cd(){  cd $(matplotlib-dir); }
matplotlib-mate(){ mate $(matplotlib-dir) ; }
matplotlib-get(){
   local dir=$(dirname $(matplotlib-dir)) &&  mkdir -p $dir && cd $dir
   svn co https://matplotlib.svn.sourceforge.net/svnroot/matplotlib/trunk/matplotlib matplotlib
}

matplotlib-versions(){
   python -c "import numpy as _ ; print _.__version__ "
   pkg-config libpng --modversion --libs --cflags
   python -c "import _tkinter"
   python -c "import Tkinter"
   #python -c "import gtk"
}

matplotlib-update(){
   svn up $(matplotlib-dir)
}

matplotlib-configdir(){  python -c "import matplotlib ; print matplotlib.get_configdir() " ; }
matplotlib-installdir(){ python -c "import matplotlib ; print matplotlib.__file__ " ; }

matplotlib-clean(){
  local msg="# === $FUNCNAME :"
  echo $msg pipe this to sh if you are happy with the deletions
  echo rm -rf $(matplotlib-configdir)
  echo rm -rf $(matplotlib-dir)/build
  echo rm -rf $(python-site)/matplotlib*

  ## TODO : add a perl -pi to cut out the line from easy-install.pth 
}


matplotlib-build(){
   matplotlib-cd
   python setupegg.py develop
}

matplotlib-test-(){ cat << EOT
from pylab import *
plot([1,2,3])
show()
EOT
}

matplotlib-test(){
  $FUNCNAME- | python - --verbose-helpful 
}



