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


Need to reinstall following move to CUDA 7.0
----------------------------------------------

::

    simon:~ blyth$ cu
    Traceback (most recent call last):
      File "/Users/blyth/env/bin/cuda_info.py", line 4, in <module>
        main()
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/env/cuda/cuda_info.py", line 98, in main
        import pycuda.driver as drv
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/driver.py", line 2, in <module>
        from pycuda._driver import *
    ImportError: dlopen(/usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/_driver.so, 2): Library not loaded: @rpath/libcurand.5.5.dylib
      Referenced from: /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/_driver.so
      Reason: image not found
    simon:~ blyth$ 




Actual Source used by Chroma ?
--------------------------------

::

    cd $VIRTUAL_ENV
    (chroma_env)delta:chroma_env blyth$ find . -name pycuda -type d
    ./build/build_pycuda/pycuda
    ./build/build_pycuda/pycuda/build/lib.macosx-10.9-x86_64-2.7/pycuda
    ./build/build_pycuda/pycuda/pycuda
    ./lib/python2.7/site-packages/pycuda


Multiple GPUs
--------------

I see there are complications
to getting pycuda to work with multiple GPUs:

* http://wiki.tiker.net/PyCuda/FrequentlyAskedQuestions#How_about_multiple_GPUs.3F
* http://wiki.tiker.net/PyCuda/FrequentlyAskedQuestions#threading

It may be more convenient to call the Chroma CUDA  C kernels 
directly in C/C++ rather than using it via PyCUDA ?

I wonder will performance scale with GPUs ?

4*5  ~ 20 in GFLOPS
4*8 ~  32 in cores



Debug/trace switches ?
------------------------

* http://git.tiker.net/pycuda.git/blob_plain/HEAD:/pycuda/driver.py


Syslog Messages
----------------

* http://lists.tiker.net/pipermail/pycuda/2013-June/004357.html

Profiling
----------

* http://lists.tiker.net/pipermail/pycuda/2012-November/004216.html


Interpreting PyCUDA errors
---------------------------

* http://lists.tiker.net/pipermail/pycuda/2014-July/004636.html

::

    > pycuda._driver.LogicError: cuMemcpyDtoH failed: invalid/unknown error code
    > PyCUDA WARNING: a clean-up operation failed (dead context maybe?)
    > cuModuleUnload failed: invalid/unknown error code

    This means your context went away while PyCUDA was still talking to
    it. This will happen most often if you perform some invalid operation
    (such as access out-of-bounds memory in a kernel). In this case, the
    cuMemcpyDtoH operation could be at fault.



* http://lists.tiker.net/pipermail/pycuda/2012-September/004106.html

::

    > pycuda._driver.LaunchError: cuMemcpyDtoH failed: launch failed
    > PyCUDA WARNING: a clean-up operation failed (dead context maybe?)
    > cuMemFree failed: launch failed
    >
    > what would be the reason ?

    It means that the operation before that (kernel execution) crashed and
    invalidated the context. Most probably, there was some read/write
    operation to the wrong memory address. You will have to look inside
    the kernel that gets executed inside gpu_devdeveigenvalues(), or
    contact its developer (do not forget to provide minimal working
    example that reproduces the bug).



* http://lists.tiker.net/pipermail/pycuda/2011-August/003335.html


::

    > I'm getting an out-of-resources error when trying to launch a CUDA
    > kernel (through PyCUDA), and I'm wondering if it's possible to get the
    > system to tell me which resource it is that I'm short on. Obviously
    > the system knows what resource has been exhausted, I just want to
    > query that as well.

    What it likely means to say is that your kernel caused a segmentation
    fault. Check the output of 'dmesg'. If it says something like 'Nv Xid
    13' (from memory, may be wrong--the 13 is what says 'segfault', I
    think), then that's what it is. Have you tried chopping your kernel down
    to almost-nothing? Bisected?




Trace pycuda slowdown via looking into ptx 
--------------------------------------------

* http://lists.tiker.net/pipermail/pycuda/2012-April/003755.html




PyCUDA Interop With Cython C code
-----------------------------------

* http://lists.tiker.net/pipermail/pycuda/2013-November/004465.html


PyCUDA Interop with Boost-Python code
--------------------------------------

* http://lists.tiker.net/pipermail/pycuda/2009-December/001993.html





Garbage Collection 
--------------------

* http://lists.tiker.net/pipermail/pycuda/2014-July/004627.html

* http://lists.tiker.net/pipermail/pycuda/2011-October/003461.html


Passing Numpy Structured Array to PyCUDA
------------------------------------------


* http://lists.tiker.net/pipermail/pycuda/2011-July/003310.html

* https://gist.github.com/inducer/88ac86874112b0e126ce



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
