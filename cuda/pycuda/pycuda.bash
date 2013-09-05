# === func-gen- : cuda/pycuda/pycuda fgp cuda/pycuda/pycuda.bash fgn pycuda fgh cuda/pycuda
pycuda-src(){      echo cuda/pycuda/pycuda.bash ; }
pycuda-source(){   echo ${BASH_SOURCE:-$(env-home)/$(pycuda-src)} ; }
pycuda-vi(){       vi $(pycuda-source) ; }
pycuda-env(){      elocal- ; }
pycuda-usage(){ cat << EOU

PYCUDA
=======

* http://documen.tician.de/pycuda/

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




EOU
}
pycuda-dir(){ echo $(local-base)/env/cuda/pycuda/cuda/pycuda-pycuda ; }
pycuda-cd(){  cd $(pycuda-dir); }
pycuda-mate(){ mate $(pycuda-dir) ; }
pycuda-get(){
   local dir=$(dirname $(pycuda-dir)) &&  mkdir -p $dir && cd $dir

}
