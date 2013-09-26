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
