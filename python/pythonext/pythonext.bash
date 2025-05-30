# === func-gen- : python/pythonext/pythonext fgp python/pythonext/pythonext.bash fgn pythonext fgh python/pythonext
pythonext-src(){      echo python/pythonext/pythonext.bash ; }
pythonext-source(){   echo ${BASH_SOURCE:-$(env-home)/$(pythonext-src)} ; }
pythonext-vi(){       vi $(pythonext-source) ; }
pythonext-env(){      elocal- ; }
pythonext-usage(){ cat << EOU

Python Extensions
==================

* https://docs.python.org/2/extending/extending.html
* http://dan.iel.fm/posts/python-c-extensions/


Numpy Array Handling Extensions
--------------------------------

* http://wiki.scipy.org/Cookbook/C_Extensions/NumPy_arrays
* http://docs.scipy.org/doc/numpy/user/c-info.how-to-extend.html
* http://blog.enthought.com/python/numpy-arrays-with-pre-allocated-memory/

npar : pull numpy ndarray out of sqlite3 queries 
--------------------------------------------------

#. build with *pythonext-build* 

   * installs into /usr/local/env/python/lib *pythonext-libdir*

#. hook that up to the chroma virtual python by adding 
   that directory to /usr/local/env/chroma_env/lib/python2.7/site-packages/easy-install.pth

#. import into ipython using *ipython-edit* adding to execlines
   *from _npar import npar as q*


ipython.sh::

    Python profile: g4dae

    In [1]: a = q("select oid, nwork, tottime from test ;")
    npar: envvar SQLITE3_DATABASE:/usr/local/env/nuwa/mocknuwa.db ncol 3 nrow 48 type f  fbufmax 1000  

    In [2]: a
    Out[2]: 
    array([[    1.   ,   445.   ,     0.07 ],
           [    2.   ,    24.   ,     0.013],
           [    3.   ,  1888.   ,     0.222],




Building
---------

::

    delta:src blyth$ pythonext-build
    running build_ext
    building '_chi2' extension
    C compiler: /usr/bin/clang -fno-strict-aliasing -fno-common -dynamic -pipe -Os -fwrapv -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes

    creating build
    creating build/temp.macosx-10.9-x86_64-2.7
    compile options: '-I/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/include 
                      -I/opt/local/Library/Frameworks/Python.framework/Versions/2.7/include/python2.7 -c'
    clang: chi2.c
    clang: _chi2.c

    In file included from _chi2.c:2:
    In file included from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/include/numpy/arrayobject.h:4:
    In file included from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/include/numpy/ndarrayobject.h:17:
    In file included from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/include/numpy/ndarraytypes.h:1760:
    /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2:
        warning: "Using deprecated NumPy API, disable it by #defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-W#warnings]
    1 warning generated.


    /usr/bin/clang -bundle 
                   -undefined dynamic_lookup 
                   -L/opt/local/lib 
                   -Wl,-headerpad_max_install_names 
                   -L/opt/local/lib/db46 
                    build/temp.macosx-10.9-x86_64-2.7/_chi2.o 
                    build/temp.macosx-10.9-x86_64-2.7/chi2.o 
                    -o /usr/local/env/python/pythonext/chi2/_chi2.so



::

    delta:~ blyth$ port provides /opt/local/lib/db46/libdb.a
    /opt/local/lib/db46/libdb.a is provided by: db46

    delta:~ blyth$ port info db46
    db46 @4.6.21_9 (databases)
    Sub-ports:            db46-java
    Variants:             compat185, java, tcl, universal

    Description:          Version 4.6 of the Berkeley Data Base library which offers (key/value) storage with optional concurrent access or transactions interface. This port will install the
                          AES (American Encryption Standard) enabled version.
    Homepage:             http://www.oracle.com/us/products/database/berkeley-db/db/overview/index.html

    Build Dependencies:   autoconf, automake, libtool
    Runtime Dependencies: db_select
    Platforms:            darwin
    License:              Sleepycat
    Maintainers:          blair@macports.org, openmaintainer@macports.org

    delta:chi2 blyth$ port rdependents db46 | grep py27 | wc -l               ## BDB heavily used by python, including numpy
          37





EOU
}

pythonext-name(){
   #echo chi2 
   echo ${PYTHONEXT_NAME:-npar}
}

pythonext-dir(){ 
    case $(pythonext-name) in 
       chi2) pythonext-idir ;;
       npar) pythonext-sdir ;;
    esac
}

pythonext-idir(){ echo $(local-base)/env/python/pythonext/$(pythonext-name) ; }
pythonext-sdir(){ echo $(env-home)/python/pythonext/$(pythonext-name) ; }
pythonext-cd(){  cd $(pythonext-dir); }
pythonext-scd(){  cd $(pythonext-sdir); }
pythonext-get(){
   local dir=$(dirname $(pythonext-dir)) &&  mkdir -p $dir && cd $dir
   case $(pythonext-name) in
     chi2) git clone git://gist.github.com/3247796.git chi2  ;;
     npar) echo -n ;;
   esac
}

pythonext-libdir-ls(){ ls -l $(pythonext-libdir) ; }
pythonext-libdir(){ echo $LOCAL_BASE/env/python/lib ; }
pythonext-tmpdir(){ echo /tmp/env/python/pythonext/$(pythonext-name) ; }
pythonext-build(){
   pythonext-cd
   python setup.py build_ext -g --build-lib $(pythonext-libdir) --build-temp $(pythonext-tmpdir)
   pythonext-check
}
pythonext-import(){
   local name=$(pythonext-name)
   PYTHONPATH=$(pythonext-libdir) python -c "import _${name}"
}
pythonext-check(){
   PYTHONPATH=$(pythonext-libdir) $LLDB python $(pythonext-sdir)/check.py 
}

pythonext-lldb(){
   LLDB=lldb pythonext-check
}

