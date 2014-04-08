# === func-gen- : cuda/pycuda/pycuda fgp cuda/pycuda/pycuda.bash fgn pycuda fgh cuda/pycuda
pycuda-src(){      echo cuda/pycuda/pycuda.bash ; }
pycuda-source(){   echo ${BASH_SOURCE:-$(env-home)/$(pycuda-src)} ; }
pycuda-vi(){       vi $(pycuda-source) ; }
pycuda-env(){      elocal- ; }
pycuda-usage(){ cat << EOU

PYCUDA
=======

* http://documen.tician.de/pycuda/
* http://mathema.tician.de/software/pycuda/


Installs
---------

Installed on D as dependency of chroma, see :doc:`/chroma/chroma`


Sources
--------

* http://git.tiker.net/pycuda.git

::

    simon:env blyth$ pycuda-get
    Cloning into 'pycuda'...
    Submodule 'bpl-subset' (https://github.com/inducer/bpl-subset) registered for path 'bpl-subset'
    Submodule 'pycuda/compyte' (https://github.com/inducer/compyte) registered for path 'pycuda/compyte'
    Cloning into 'bpl-subset'...
    Submodule path 'bpl-subset': checked out 'e7c5f5131daca6298b5e8aa48d06e7ecffec2ffa'
    Cloning into 'pycuda/compyte'...
    Submodule path 'pycuda/compyte': checked out '6ccb955b0d38faffeb7dd5bf913e6bedf46ee226'


Tags
~~~~~

* http://git.tiker.net/?p=pycuda.git;a=summary

::

    tags
    8 months ago    v2013.1.1    
    8 months ago    v2013.1     
    20 months ago   v2012.1    
    2 years ago v2011.2.2     
    2 years ago v2011.2.1    
    2 years ago v2011.2     
    3 years ago v0.94.2    
    3 years ago v0.94.1   
    3 years ago v0.94    
    3 years ago v0.94rc   
    4 years ago v0.93.1rc2  
    4 years ago v0.93      
    4 years ago v0.93rc4  
    4 years ago v0.93rc3   
    4 years ago v0.93rc2  
    4 years ago v0.93rc1 
    ...


Linux Install
-------------

* http://wiki.tiker.net/PyCuda/Installation/Linux

Linux Ubuntu Install
--------------------

* http://wiki.tiker.net/PyCuda/Installation/Linux/Ubuntu

Inspect Versions
~~~~~~~~~~~~~~~~~

::

    aracity@aracity-desktop:~$ lspci | grep VGA
    01:00.0 VGA compatible controller: nVidia Corporation GT200 [GTX260-216] (rev a1)
    aracity@aracity-desktop:~$ lsb_release -a 2> /dev/null | grep Release
    Release:        9.04
    aracity@aracity-desktop:~$ g++ --version | head -1
    g++ (Ubuntu 4.3.3-5ubuntu4) 4.3.3
    aracity@aracity-desktop:~$ gcc --version | head -1
    gcc (Ubuntu 4.3.3-5ubuntu4) 4.3.3


Prerequisites
---------------

Assume a local user build is required, ie no system access.

Argh, its a major pain to live without package management maybe 
worthwhile to attack the old chrootnut :doc:`/linux/non-root-package-management`


python 
~~~~~~

::

    python-name  # check than an appropriate version is slated to be locally source built, eg 2.6.8 
    pythonbuild-get
    pythonbuild-configure
    pythonbuild-install
    python-        # environment to access the local python

::

    -bash-3.2$ which python
    /usr/bin/python
    -bash-3.2$ python-
    -bash-3.2$ which python
    ~/local/python/Python-2.6.8/bin/python
    -bash-3.2$ python -V
    Python 2.6.8


mavericks
~~~~~~~~~~~

* http://wiki.tiker.net/PyCuda/FrequentlyAskedQuestions

CUDA is composed of two APIs:

#. A low-level API called the CUDA driver API,
#. A higher-level API called the CUDA runtime API that is implemented on top of the CUDA driver API. 

These APIs are mutually exclusive: **An application should use either one or the other.**
PyCUDA is based on the driver API. 
CUBLAS uses the high-level API. 
One can violate this rule without crashing immediately. But sketchy stuff does happen. 
Instead, for BLAS-1 operations, PyCUDA comes with a class called GPUArray that essentially 
reimplements that part of CUBLAS.


* http://documen.tician.de/pycuda/


download examples from wiki
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    (chroma_env)delta:examples blyth$ pwd
    /usr/local/env/chroma_env/build/build_pycuda/pycuda/examples
    (chroma_env)delta:examples blyth$ python download-examples-from-wiki.py 
    downloading  wiki examples from http://wiki.tiker.net/PyCuda/Examples to wiki-examples/...
    fetching page list...
    PyCuda/Examples/SobelFilter
    ...
    PyCuda/Examples/GlInterop
    PyCuda/Examples/LightField_3D_viewer
    ...
    PyCuda/Examples/DemoMetaCodepy
    Error when processing PyCuda/Examples/DemoMetaCodepy: 'NoneType' object has no attribute 'group'
    Traceback (most recent call last):
      File "download-examples-from-wiki.py", line 29, in <module>
        code = match.group(1)
    AttributeError: 'NoneType' object has no attribute 'group'
    PyCuda/Examples/ArithmeticExample
    PyCuda/Examples/Mandelbrot
    ...




EOU
}
#pycuda-dir(){ echo $VIRTUAL_ENV/lib/python2.7/site-packages/pycuda ; }
pycuda-dir(){ echo $VIRTUAL_ENV/build/build_pycuda/pycuda ; }
pycuda-cd(){  cd $(pycuda-dir)/$1; }
pycuda-mate(){ mate $(pycuda-dir) ; }
pycuda-get(){
   local dir=$(dirname $(pycuda-dir)) &&  mkdir -p $dir && cd $dir

  [ ! -d "pycuda" ] &&  git clone --recursive http://git.tiker.net/trees/pycuda.git
  # not using the git version, used pip install via chroma functions
}


pycuda-examples-cd(){
   pycuda-cd examples/wiki-examples
}
