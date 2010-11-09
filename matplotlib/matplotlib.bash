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

    sudo yum install tcl-devel
    sudo yum install tk-devel

 After which rebuild python  (just a few min partial build):

   pythonbuild-
   pythonbuild-configure
   pythonbuild-install
   python -c "import Tkinter"


 Propagate to matplotlib by cleaning and rebuilding as described above

   Test it worked ...
        ipython -pylab
        > figure()      ## should popup a GUI window 
 


EOU
}
matplotlib-dir(){ echo $(local-base)/env/matplotlib/matplotlib ; }
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
}

matplotlib-update(){
   svn up $(matplotlib-dir)
}

matplotlib-build(){
   matplotlib-cd
   python setupegg.py develop
}


