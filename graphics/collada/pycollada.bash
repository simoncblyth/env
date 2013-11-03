# === func-gen- : graphics/collada/pycollada fgp graphics/collada/pycollada.bash fgn pycollada fgh graphics/collada
pycollada-src(){      echo graphics/collada/pycollada.bash ; }
pycollada-source(){   echo ${BASH_SOURCE:-$(env-home)/$(pycollada-src)} ; }
pycollada-vi(){       vi $(pycollada-source) ; }
pycollada-env(){      elocal- ; }
pycollada-usage(){ cat << EOU

PYCOLLADA
==========

* http://pycollada.github.io/creating.html
* http://www.khronos.org/collada/


REQUIREMENTS
-------------

* py26+
* numpy
* unittest2
* python-dateutil 1.5(py26+)  2.0(py3)


INSTALLS
---------


N
~~




C2
~~~

#. system python 2.3.4 to old to attempt, source python 2.5.6, would need access to my py25compat git branch
#. machine is currently network blocked, have to access via H

G
~~

With py25 many errors at install, but build seemed ok.  Select py26 and try again::

    simon:~ blyth$ sudo port select --list python
    Available versions for python:
            none
            python25 (active)
            python25-apple
            python26
            python27

    simon:~ blyth$ sudo port select --set python python26
    Selecting 'python26' for 'python' succeeded. 'python26' is now active.

    simon:pycollada blyth$ sudo port select --set ipython ipython26
    Selecting 'ipython26' for 'ipython' succeeded. 'ipython26' is now active.


    simon:~ blyth$ python -V
    Python 2.6.8


G system python
~~~~~~~~~~~~~~~~~~

Use of meshtools/pycollada/panda3d/Cg.framework for viewing collada docs
provides a reason to backport pycollada to system py2.5.1 on OSX. Hence::

    simon:pycollada blyth$ git checkout -b "py25compat"
    Switched to a new branch 'py25compat'

    simon:pycollada blyth$ git branch --list 
      master
    * py25compat

Switching "as" to "comma" and adding a few "from future import with_statement" 
succeeds to get pycollada to install into system py25

::

     perl -pi -e 's,(except )(\S*)( as )(ex:),$1$2\,$4,' *.py 

     sudo ppython setup.py install




EOU
}
pycollada-dir(){ echo $(local-base)/env/graphics/collada/pycollada ; }
pycollada-cd(){  cd $(pycollada-dir); }
pycollada-scd(){  cd $(env-home)/graphics/collada/pycollada ; }
pycollada-mate(){ mate $(pycollada-dir) ; }
pycollada-get(){
   local dir=$(dirname $(pycollada-dir)) &&  mkdir -p $dir && cd $dir
   [ ! -d pycollada ] && git clone git://github.com/pycollada/pycollada.git pycollada 
}


pycollada-daeview-orig(){
   local cmd="$(pycollada-dir)/examples/daeview/daeview.py $(env-home)/graphics/collada/pycollada/test.dae "
   echo $cmd
   eval $cmd
}

pycollada-view(){
  $(env-home)/graphics/collada/daeview/daeview.py $1
}

pycollada-daeview(){
   cd $(env-home)/graphics/collada/daeview
   local def=$(env-home)/graphics/collada/pycollada/subcopy.dae
   local dae=${1:-$def}
   #local cmd="./daeview.py $dae"
   local cmd="$(env-home)/graphics/collada/daeview/daeview.py $dae"
   echo $cmd
   eval $cmd
}


pycollada-build(){
   pycollada-cd
   python setup.py build 
}

pycollada-install(){
   pycollada-cd
   sudo python setup.py install
}

pycollada-wipe(){
   pycollada-cd
   sudo rm -rf build dist pycollada.egg-info
}

pycollada-check(){
    which python
    python -V 
    python -c "import sys ; print sys.version_info "
    python -c "import numpy"
}
