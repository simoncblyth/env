# === func-gen- : numpy/npyreader fgp numpy/npyreader.bash fgn npyreader fgh numpy
npyreader-src(){      echo numpy/npyreader.bash ; }
npyreader-source(){   echo ${BASH_SOURCE:-$(env-home)/$(npyreader-src)} ; }
npyreader-vi(){       vi $(npyreader-source) ; }
npyreader-env(){      elocal- ; }
npyreader-usage(){ cat << EOU

NPYREADER
============

Special cased reader for npy files in C.

* http://jcastellssala.wordpress.com/2014/02/01/npy-in-c/
* http://sourceforge.net/projects/kxtells.u/files/

npy file format
----------------

A Simple File Format for NumPy Arrays

* https://github.com/numpy/numpy/blob/master/doc/neps/npy-format.rst
* http://stackoverflow.com/questions/4090080/what-is-the-way-data-is-stored-in-npy

  * its about as simple as it could possibly be
  * fixed position first 10 bytes: magic(6), version bytes(2), header size(2)
  * variable header size with type info in form of a python dict string repr  
  * remainder array data, get size by seeking on file size and subtracting off fixed + header size 


from ipython
-------------

::

    In [1]: import numpy as np

    In [2]: a = np.load("1d_data.npy")

    In [3]: a
    Out[3]: 
    array([-0.06278012, -0.04804549, -0.06925336, ..., -0.04754395,
           -0.05315629, -0.05407996], dtype=float32)

    In [4]: a.shape
    Out[4]: (3000,)

    In [6]: b = np.load("2d_data.npy")

    In [7]: b
    Out[7]: 
    array([[ 0.12310834, -0.01396761, -0.21924776, ..., -0.06124042,
             0.09163436, -0.12671494],
           [-0.08736823, -0.00782842,  0.04040892, ...,  0.01441374,
            -0.0650332 ,  0.13787675],
           [-0.12703118, -0.11481609, -0.03706783, ..., -0.07172504,
             0.10714594,  0.06278787],
           ..., 
           [ 0.05658171,  0.07645642, -0.06426196, ..., -0.06957918,
             0.0310857 , -0.04734502],
           [ 0.16576487,  0.05594297,  0.0491025 , ...,  0.14608166,
            -0.15042561, -0.04379739],
           [-0.12791227, -0.14804763,  0.09068686, ..., -0.13774954,
            -0.10448983,  0.06895938]], dtype=float32)

    In [8]: b.shape
    Out[8]: (51, 3000)




check
------

#. after creating test_utils.h and running the check from top level, successful test 


::

    delta:npyreader-0.01 blyth$ make check
    Making check in src/lib
    make[1]: Nothing to be done for `check'.
    Making check in src/test
    /Applications/Xcode.app/Contents/Developer/usr/bin/make  test_npyreader_1d test_npyreader_2d
    gcc -DHAVE_CONFIG_H -I. -I../..    -I../lib -g -O2 -MT test_npyreader_1d-test_npyreader_1d.o -MD -MP -MF .deps/test_npyreader_1d-test_npyreader_1d.Tpo -c -o test_npyreader_1d-test_npyreader_1d.o `test -f 'test_npyreader_1d.c' || echo './'`test_npyreader_1d.c
    ...
    /Applications/Xcode.app/Contents/Developer/usr/bin/make  check-TESTS
    PASS: test_npyreader_1d
    PASS: test_npyreader_2d
    make[4]: Nothing to be done for `all'.
    ============================================================================
    Testsuite summary for npyreader 0.01
    ============================================================================
    #TOTAL: 2
    # PASS:  2
    # SKIP:  0
    # XFAIL: 0
    # FAIL:  0
    # XPASS: 0
    # ERROR: 0
    ============================================================================
    make[1]: Nothing to be done for `check-am'.
    delta:npyreader-0.01 blyth$ 






EOU
}
npyreader-name(){ echo npyreader-0.01 ; }
npyreader-dir(){    echo $(local-base)/env/numpy/$(npyreader-name) ; }
npyreader-prefix(){ echo $(local-base)/env/numpy/npyreader ; }
npyreader-libdir(){ echo $(npyreader-dir)/src/lib ; }
npyreader-incdir(){ echo $(npyreader-dir)/src/lib ; }
npyreader-libname(){ echo npyreader ; }
npyreader-cd(){  cd $(npyreader-dir); }
npyreader-mate(){ mate $(npyreader-dir) ; }
npyreader-get(){
   local dir=$(dirname $(npyreader-dir)) &&  mkdir -p $dir && cd $dir

   local nam=$(npyreader-name)
   local tgz=$nam.tar.gz

   [ ! -f "$tgz" ] && curl -L -O http://downloads.sourceforge.net/project/kxtells.u/$tgz
   [ ! -d "$nam" ] && tar zxvf $tgz

}

npyreader-build(){

   npyreader-cd
  ./configure --prefix=$(npyreader-prefix)

   make 
   make install

}

