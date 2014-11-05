# === func-gen- : zeromq/pyzmq/pyzmq fgp zeromq/pyzmq/pyzmq.bash fgn pyzmq fgh zeromq/pyzmq
pyzmq-src(){      echo zeromq/pyzmq/pyzmq.bash ; }
pyzmq-source(){   echo ${BASH_SOURCE:-$(env-home)/$(pyzmq-src)} ; }
pyzmq-vi(){       vi $(pyzmq-source) ; }
pyzmq-usage(){ cat << EOU

PYZMQ
=======

* PyZMQ works with Python 3 (>= 3.2), and Python 2 (>= 2.6)

* http://zeromq.org/bindings:python
* http://zeromq.github.io/pyzmq/

* https://github.com/zeromq/pyzmq


fix memory leak in zero-copy allocation 
https://github.com/zeromq/pyzmq/pull/517


Recommended install from pypi
--------------------------------

Attempt install into nuwa python::

    [blyth@belle7 ~]$ which easy_install
    /data1/env/local/dyb/external/setuptools/0.6c11_python2.7/i686-slc5-gcc41-dbg/bin/easy_install
    [blyth@belle7 ~]$ 
    [blyth@belle7 ~]$ 
    [blyth@belle7 ~]$ easy_install pyzmq
    Searching for pyzmq
    Reading http://pypi.python.org/simple/pyzmq/
    Reading http://github.com/zeromq/pyzmq
    Reading http://github.com/zeromq/pyzmq/downloads
    Reading http://github.com/zeromq/pyzmq/releases
    Best match: pyzmq 14.3.1
    Downloading https://pypi.python.org/packages/source/p/pyzmq/pyzmq-14.3.1.zip#md5=b6323c14774ab5

    Did not find libzmq via pkg-config:
    Package libzmq was not found in the pkg-config search path.
    Perhaps you should add the directory containing `libzmq.pc'
    to the PKG_CONFIG_PATH environment variable
    No package 'libzmq' found
     

    [blyth@belle7 dyb]$ cat /data1/env/local/dyb/external/zmq/4.0.4/i686-slc5-gcc41-dbg/lib/pkgconfig/libzmq.pc 
    prefix=/data1/env/local/dyb/NuWa-trunk/../external/zmq/4.0.4/i686-slc5-gcc41-dbg
    exec_prefix=${prefix}
    libdir=${exec_prefix}/lib
    includedir=${prefix}/include

    Name: libzmq
    Description: 0MQ c++ library
    Version: 4.0.4
    Libs: -L${libdir} -lzmq
    Cflags: -I${includedir}
    [blyth@belle7 dyb]$ 


::

    [blyth@belle7 dyb]$ fenv
    [blyth@belle7 dyb]$ PKG_CONFIG_PATH=/data1/env/local/dyb/external/zmq/4.0.4/i686-slc5-gcc41-dbg/lib/pkgconfig easy_install pyzmq
    Searching for pyzmq
    Reading http://pypi.python.org/simple/pyzmq/
    Reading http://github.com/zeromq/pyzmq
    Reading http://github.com/zeromq/pyzmq/downloads
    Reading http://github.com/zeromq/pyzmq/releases
    Best match: pyzmq 14.3.1
    Downloading https://pypi.python.org/packages/source/p/pyzmq/pyzmq-14.3.1.zip#md5=b6323c14774ab5bd401112b259bf70be
    Processing pyzmq-14.3.1.zip
    Running pyzmq-14.3.1/setup.py -q bdist_egg --dist-dir /tmp/easy_install-MWueRn/pyzmq-14.3.1/egg-dist-tmp-wtsFfj
    no previously-included directories found matching 'docs/build'
    no previously-included directories found matching 'docs/gh-pages'
    warning: no previously-included files found matching 'bundled/zeromq/src/Makefile*'
    warning: no previously-included files found matching 'bundled/zeromq/src/platform.hpp'
    warning: no previously-included files found matching 'setup.cfg'
    warning: no previously-included files found matching 'zmq/libzmq*'
    warning: no previously-included files matching '__pycache__/*' found anywhere in distribution
    warning: no previously-included files matching '.deps/*' found anywhere in distribution
    warning: no previously-included files matching '*.so' found anywhere in distribution
    warning: no previously-included files matching '*.pyd' found anywhere in distribution
    warning: no previously-included files matching '.git*' found anywhere in distribution
    warning: no previously-included files matching '.DS_Store' found anywhere in distribution
    warning: no previously-included files matching '.mailmap' found anywhere in distribution
    warning: no previously-included files matching 'Makefile.am' found anywhere in distribution
    warning: no previously-included files matching 'Makefile.in' found anywhere in distribution
    Using zmq-prefix /data1/env/local/dyb/NuWa-trunk/../external/zmq/4.0.4/i686-slc5-gcc41-dbg (found via pkg-config).
    build/temp.linux-i686-2.7/scratch/check_sys_un.c: In function 'main':
    build/temp.linux-i686-2.7/scratch/check_sys_un.c:6: warning: format '%lu' expects type 'long unsigned int', but argument 2 has type 'unsigned int'
    ************************************************
    Configure: Autodetecting ZMQ settings...
        Custom ZMQ dir:       
    build/temp.linux-i686-2.7/scratch/tmp/easy_install-MWueRn/pyzmq-14.3.1/temp/timer_createHs6RQh.o: In function `main':
    timer_createHs6RQh.c:(.text+0x12): undefined reference to `timer_create'
    collect2: ld returned 1 exit status
        ZMQ version detected: 4.0.4
    ************************************************
    zip_safe flag not set; analyzing archive contents...
    zmq.__init__: module references __file__
    zmq.backend.cffi._cffi: module references __file__
    Adding pyzmq 14.3.1 to easy-install.pth file

    Installed /data1/env/local/dyb/external/Python/2.7/i686-slc5-gcc41-dbg/lib/python2.7/site-packages/pyzmq-14.3.1-py2.7-linux-i686.egg
    Processing dependencies for pyzmq
    Finished processing dependencies for pyzmq
    [blyth@belle7 dyb]$ 






EOU
}
pyzmq-dir(){ echo $(env-home)/zeromq/pyzmq ; }
pyzmq-cd(){  cd $(pyzmq-dir); }
pyzmq-mate(){ mate $(pyzmq-dir) ; }

pyzmq-env(){    
    elocal- 
    chroma-   # for the right python
    zmq-  # client/broker/worker config
}

pyzmq-operator(){ python $(pyzmq-dir)/zmq_operator.py $* ; }

pyzmq-broker(){ FRONTEND=tcp://*:$(zmq-frontend-port) BACKEND=tcp://*:$(zmq-backend-port) pyzmq-operator broker ; }
pyzmq-client(){ FRONTEND=tcp://$(zmq-broker-host):$(zmq-frontend-port) pyzmq-operator client ; }
pyzmq-worker(){ BACKEND=tcp://$(zmq-broker-host):$(zmq-backend-port)   pyzmq-operator worker ; }
pyzmq-mirror(){ BACKEND=tcp://$(zmq-broker-host):$(zmq-backend-port)   pyzmq-operator mirror ; }
pyzmq-npymirror(){ BACKEND=tcp://$(zmq-broker-host):$(zmq-backend-port)   pyzmq-operator npymirror ; }




