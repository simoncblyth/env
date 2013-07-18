# === func-gen- : plot/svgplotlib fgp plot/svgplotlib.bash fgn svgplotlib fgh plot
svgplotlib-src(){      echo plot/svgplotlib.bash ; }
svgplotlib-source(){   echo ${BASH_SOURCE:-$(env-home)/$(svgplotlib-src)} ; }
svgplotlib-vi(){       vi $(svgplotlib-source) ; }
svgplotlib-env(){      elocal- ; }
svgplotlib-usage(){ cat << EOU

svgplotlib
============

Python package to create SVG charts and graphs.

* http://code.google.com/p/svgplotlib/
* at 0.2 nice and lightweight, looking ahead in trunk gets heavy 
* Mixed licence: BSD, LGPL, GPL

.. warning:: not installed, stuck at pyparsing 


0.2 issues 
-----------

freetype build issue (py25)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At build::

   simon:svgplotlib-0.2 blyth$ python setup.py build

Get freetype headers missing::

    svgplotlib/freetype.c:239:30: error: freetype/fttypes.h: No such file or directory
    ...

Resolved by adding to include_dirs in setup.py::

    include_dirs = ['svgplotlib','svgplotlib/include', '/opt/local/include/freetype2', '/opt/local/include'],

Try::

    simon:svgplotlib-0.2 blyth$ sudo port install freetype

runtime requires 
~~~~~~~~~~~~~~~~~~~~~~~~~

#. collections.namedtuple, so needs at least py26
#. pyparsing, not in macports

G py26 build/install
~~~~~~~~~~~~~~~~~~~~~~~

::

    sudo port install py26-cython

    simon:svgplotlib-0.2 blyth$ python2.6 setup.py build
    simon:svgplotlib-0.2 blyth$ sudo python2.6 setup.py install

EOU
}
svgplotlib-dir(){ echo $(local-base)/env/plot/$(svgplotlib-name) ; }
svgplotlib-cd(){  cd $(svgplotlib-dir); }
svgplotlib-mate(){ mate $(svgplotlib-dir) ; }
svgplotlib-name(){ echo svgplotlib-0.2 ; }
#svgplotlib-name(){ echo svgplotlib ; }
svgplotlib-get(){
   local dir=$(dirname $(svgplotlib-dir)) &&  mkdir -p $dir && cd $dir
   local nam=$(svgplotlib-name)
   local tgz="$nam.tar.gz"
   local url="http://svgplotlib.googlecode.com/files/$tgz"
   [ ! -f "$tgz" ] && curl -L -O "$url"
   [ ! -d "$nam" ] && tar zxvf "$tgz"
}

svgplotlib-co(){
  local dir=$(dirname $(svgplotlib-dir)) &&  mkdir -p $dir && cd $dir
  svn checkout http://svgplotlib.googlecode.com/svn/trunk/ svgplotlib
}
