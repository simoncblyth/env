# === func-gen- : matplotlib/matplotlib fgp matplotlib/matplotlib.bash fgn matplotlib fgh matplotlib
matplotlib-src(){      echo matplotlib/matplotlib.bash ; }
matplotlib-source(){   echo ${BASH_SOURCE:-$(env-home)/$(matplotlib-src)} ; }
matplotlib-vi(){       vi $(matplotlib-source) ; }
matplotlib-env(){      elocal- ; }
matplotlib-usage(){
  cat << EOU

Matplotlib
===========

* see mpl- for usage rather than installation notes

Version
---------

::

    delta:~ blyth$ python -c "import matplotlib as _ ; print _.__file__, _.__version__ "
    /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/matplotlib/__init__.pyc 1.3.1


controlling window position ?
--------------------------------

* https://stackoverflow.com/questions/8202228/make-matplotlib-plotting-window-pop-up-as-the-active-one_

::

    In [1]: mngr = plt.get_current_fig_manager()

    In [2]: mngr.window.setGeometry(50,100,640, 545)

    AttributeError: 'FigureManagerMac' object has no attribute 'window'

    In [3]: mngr.show()   ## brings to front 

    In [4]: mngr.num
    Out[4]: 2

    In [5]: mngr.canvas
    Out[5]: FigureCanvas object 0x1616420f0 wrapping NSView 0x7fb2106e9300

    In [16]: mngr.canvas?
    Type:        FigureCanvasMac
    String form: FigureCanvas object 0x1616420f0 wrapping NSView 0x7fb2106e9300
    File:        ~/miniconda3/lib/python3.7/site-packages/matplotlib/backends/backend_macosx.py

    In [6]: mngr.get_window_title()
    Out[6]: 'Figure 2'


OSX installation
------------------

On trying a basic plot using matplotlib installed into virtual python based on
macports python26::


	import matplotlib.pyplot as plt
	import numpy as np
	x = np.random.randn(1000)
	plt.hist( x, 20)
	plt.grid()
	plt.title(r'Normal: $\mu=%.2f, \sigma=%.2f$'%(x.mean(), x.std()))
	plt.show()

The plot appears, but get a not-a-framework warning::

	In [5]: plt.hist( x, 20)
	/usr/local/env/matplotlib/matplotlib-1.0.0/lib/matplotlib/backends/backend_macosx.py:235: 
	UserWarning: 
	  Python is not installed as a framework. 
	  The MacOSX backend may not work correctly if Python is not installed as a framework. 
	  Please see the Python documentation for more information on installing Python as a framework on Mac OS X


matplotlib.sphinxext.ipython_directive 
-----------------------------------------

 * this hails from the src install on N, on G 


::

	[blyth@belle7 sphinxext]$ pwd
	/home/blyth/rst/src/matplotlib/lib/matplotlib/sphinxext
	[blyth@belle7 sphinxext]$ ll
	total 116
	drwxrwxr-x  3 blyth blyth  4096 Jan  4 19:55 .
	drwxrwxr-x 11 blyth blyth  4096 Jan  4 19:54 ..
	-rw-rw-r--  1 blyth blyth     1 Jan  4 15:28 __init__.py
	-rw-rw-r--  1 blyth blyth   154 Jan  4 19:44 __init__.pyc
	-rw-rw-r--  1 blyth blyth  4183 Jan  4 15:28 ipython_console_highlighting.py
	-rw-rw-r--  1 blyth blyth  3266 Jan  4 19:55 ipython_console_highlighting.pyc
	-rw-rw-r--  1 blyth blyth 15656 Jan  4 15:28 ipython_directive.py
	-rw-rw-r--  1 blyth blyth 11909 Jan  4 19:54 ipython_directive.pyc
	-rw-rw-r--  1 blyth blyth  3781 Jan  4 15:28 mathmpl.py
	-rw-rw-r--  1 blyth blyth  5040 Jan  4 19:51 mathmpl.pyc
	-rw-rw-r--  1 blyth blyth  2109 Jan  4 15:28 only_directives.py
	-rw-rw-r--  1 blyth blyth  3603 Jan  4 19:44 only_directives.pyc
	-rw-rw-r--  1 blyth blyth 18563 Jan  4 15:28 plot_directive.py
	-rw-rw-r--  1 blyth blyth 15100 Jan  4 19:44 plot_directive.pyc
	drwxrwxr-x  6 blyth blyth  4096 Jan  4 15:32 .svn



installation 
--------------

Pre-requisites, 

#. numpy (my fork with a few fixes, they should be incoporated upstream now?)
#. mysql_numpy (my fork of mysql_python 1.2.3 that adds the pulling of numpy arrays from mysql query results)::

         pip install -E ~/v/docs -e git+git://github.com/scb-/numpy.git#egg=numpy                ## maybe better to use the writable access technique         
         pip install -E ~/v/docs -e git://github.com/scb-/mysql_numpy.git#egg=mysql_numpy 

Hmm "mysql_numpy" fails as setup.py is not at top level, have to::

         cd ~/v/docs/src/mysql-numpy/MySQLdb/
         python setup.py install

Then matplotlib (fails if numpy not installed)::

         pip install -E ~/v/docs -f http://downloads.sourceforge.net/project/matplotlib/matplotlib/matplotlib-1.0/matplotlib-1.0.0.tar.gz -U matplotlib

For testing and interactive dev::

         pip install -E ~/v/docs -U ipython
         pip install -E ~/v/docs -U nose

env hookup::

         cd ~/v/docs/lib/python2.6/site-packages 
         ln -sf ~/env env

ipython gotcha
          despite what "which ipython" would have you imagine, be explicit in the 
          virtual ipython invocation in order to have the full virtual sys.path 

          ~/v/docs/bin/ipython dcspmthv.py 

tracker : appears neglected

 * http://sourceforge.net/tracker/?atid=560720&group_id=80706&func=browse

primitive archive views
 
 * http://sourceforge.net/mail/?group_id=80706
 * http://sourceforge.net/mailarchive/forum.php?forum_name=matplotlib-users
  
better interface 

  * http://news.gmane.org/gmane.comp.python.matplotlib.general
  * http://news.gmane.org/gmane.comp.python.matplotlib.devel

primitive svn browser

  * http://matplotlib.svn.sourceforge.net/viewvc/matplotlib/

git mirror .. 

  * https://github.com/astraw/matplotlib#readme

huh lots of activity in maintenance branch ?

  * http://matplotlib.svn.sourceforge.net/viewvc/matplotlib/branches/v1_0_maint/lib/matplotlib/


install
~~~~~~~~~~

pypi entry for matplotlib is buggered ... causing an out of date install
workaround, use a source install::

     pip -v -E ~/v/docs install -e svn+https://matplotlib.svn.sourceforge.net/svnroot/matplotlib/trunk/matplotlib/#egg=matplotlib


interactive plotting approaches 
---------------------------------
   
Thinking about how to implement interactive plots using SVG 
brought me to matplotlib as a higher level way of proceeding.
  
I looked into  
 * http://raphaeljs.com/       (hindered by supporting IE canvas... not an SVG API)

Interactive SVG 
 * http://www.svgopen.org/2009/papers/14-Interactive_SVG_with_JSXGraph/  
      
html5 : sliders are trivial with html5 ... just a range input 
 * http://webhole.net/2010/04/24/html-5-slider-input-tutorial/

::

	<html>
	<body>
	<input type="range" min="0" max="50" value="0" step="5" onchange="showValue(this.value)" />
	<span id="range">0</span>
	<script type="text/javascript">
	function showValue(newValue)
	{
		document.getElementById("range").innerHTML=newValue;
	}
	</script>
	</body>
	</html>


svg slider ...
  * http://www.carto.net/papers/svg/gui/slider/
  * http://www.carto.net/papers/svg/gui/slider/index.svg     


EOU
}

matplotlib-issues(){ cat << EOI

  == ISSUE : pip install matplotlib ... plucks 0.91.1 (from 2007) ==

     Workaround: skip pypi index and target the tarball directly :
       pip install -f http://downloads.sourceforge.net/project/matplotlib/matplotlib/matplotlib-1.0/matplotlib-1.0.0.tar.gz matplotlib
       pip install -E ~/v/docs -f http://downloads.sourceforge.net/project/matplotlib/matplotlib/matplotlib-1.0/matplotlib-1.0.0.tar.gz -U matplotlib

     this fails if numpy is not installed ..
     so ... actually before installing matplotlib better to install my forked numpy + mysql_numpy 

         pip install -E ~/v/docs -e git+git://github.com/scb-/numpy.git#egg=numpy                ## maybe better to use the writable access technique
         pip install -E ~/v/docs -e git://github.com/scb-/mysql_numpy.git#egg=mysql_numpy 


  == ISSUE BLANK CANVAS WITH TkAgg ON C and N ... ==

     * less of an issue on N, as GTkAgg is available ...

   Suspicious commit ...
     * https://github.com/astraw/matplotlib/commit/e4927a719403a769709d92c0dac659c02931308a

   ------------------------------------------------------------------------
r8739 | efiring | 2010-10-11 04:30:37 +0800 (Mon, 11 Oct 2010) | 2 lines
Changed paths:
   M /trunk/matplotlib/lib/matplotlib/backends/backend_tkagg.py

backend_tkagg: delete dead code


    https://github.com/astraw/matplotlib/tree/trunk/lib/matplotlib/backends/
    https://matplotlib.svn.sourceforge.net/svnroot/matplotlib/branches/v1_0_maint/

 == WORKAROUND TkAgg FAILURE ===> MOVE TO GTkAgg ==


    After hard work in pygtk- (pygobject- that turned out not be needed at the requisite version era)
    succeed to build the GTkAgg backend into source python on C

         Gtk+: gtk+: 2.4.13, glib: 2.4.7, pygtk: 2.4.0, pygobject:
               [pre-pygobject]

    Now the plots reappear


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





EOI
}

matplotlib-name(){
    echo matplotlib         ## trunk 
   #echo matplotlib-1.0.0   ## tarball
}
matplotlib-update(){ svn up $(matplotlib-dir) ; }

matplotlib-dir(){ 
  local nam=$(matplotlib-name) 
  case $nam in 
         matplotlib) [ -n "$VIRTUAL_ENV" ] && echo $VIRTUAL_ENV/src/matplotlib ||  echo $(local-base)/env/matplotlib/matplotlib ;;
      matplotlib-* ) echo $(local-base)/env/matplotlib/$nam ;; 
  esac 
}
matplotlib-cd(){  cd $(matplotlib-dir); }
matplotlib-ex(){ cd $(env-home)/matplotlib/examples ; }
matplotlib-mate(){ mate $(matplotlib-dir) ; }




matplotlib-get(){
   local dir=$(dirname $(matplotlib-dir)) &&  mkdir -p $dir && cd $dir
   local nam=$(matplotlib-name) 
   if [ "$nam" == "matplotlib" ]; then
       svn co https://matplotlib.svn.sourceforge.net/svnroot/matplotlib/trunk/matplotlib matplotlib
   else  
       [ ! -f "$nam.tar.gz" ] && curl -L  -O http://downloads.sourceforge.net/project/matplotlib/matplotlib/matplotlib-1.0/$nam.tar.gz
       [ ! -d "$nam" ]        && tar zxvf $nam.tar.gz
   fi
}



matplotlib-preqs(){
   echo numpy...
   numpy-
   numpy-info

   echo libpng ...
   pkg-config libpng --modversion --libs --cflags

   echo checking _tkinter Tkinter...
   python -c "import _tkinter"
   python -c "import Tkinter"

   echo pygtk
   pygtk-
   pygtk-info
   
}


matplotlib-configdir(){  python -c "import matplotlib ; print matplotlib.get_configdir() " ; }
matplotlib-installdir(){ python -c "import matplotlib,os ; print os.path.dirname(matplotlib.__file__) " ; }
matplotlib-version(){    python -c "import matplotlib ; print matplotlib.__version__ "  ; }
matplotlib-rcpath(){     python -c "import matplotlib ; print matplotlib.matplotlib_fname() " ; }
matplotlib-edit(){       vi $(matplotlib-rcpath) ; }     
matplotlib-easy(){       grep matplotlib $(python-site)/easy-install.pth ; }


matplotlib-info(){
   cat << EOI
     version    : $(matplotlib-version)
     configdir  : $(matplotlib-configdir)
     installdir : $(matplotlib-installdir)
     rcpath     : $(matplotlib-rcpath)
     easy       : $(matplotlib-easy)

EOI
}


matplotlib-uneasy-(){
  local lib=$(matplotlib-dir)/lib
  echo perl -pi -e \"s,$lib\\n,,\" $(python-site)/easy-install.pth 
}
matplotlib-unbuild-(){
  local msg="# === $FUNCNAME :"
  echo rm -rf $(matplotlib-dir)/build
  [ ! -d "$(matplotlib-dir)/.svn" ] && echo $msg not svn checkout ... skip && return   
  svn status --no-ignore $(matplotlib-dir) | perl -p -e 's,^[I?],rm -rf,g' - 
}
matplotlib-pyclean-(){
  local msg="# === $FUNCNAME :"
  [ "$(matplotlib-version)" == "" ] && echo $msg python sees no matplotlib && return   
  echo rm -rf $(matplotlib-configdir)
  echo rm -rf $(python-site)/matplotlib*
  echo rm -rf $(python-site)/mpl_toolkits*
}
matplotlib-clean(){
   local msg="# === $FUNCNAME :"
   echo $msg pipe this to sh if you are happy with the deletions
   matplotlib-pyclean-
   matplotlib-unbuild-
   matplotlib-uneasy-
}



matplotlib-build(){
   matplotlib-cd
   #python setup.py build
   #python setup.py install
   python setupegg.py develop
}

matplotlib-sbuild(){
   matplotlib-cd
   #python setup.py build
   #python setup.py install
   sudo python setupegg.py develop
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



