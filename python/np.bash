# === func-gen- : python/np fgp python/np.bash fgn np fgh python
np-src(){      echo python/np.bash ; }
np-source(){   echo ${BASH_SOURCE:-$(env-home)/$(np-src)} ; }
np-vi(){       vi $(np-source) ; }
np-env(){      elocal- ; }
np-usage(){ cat << EOU

Plain Vanilla Usage of numpy
==============================

See also *numpy-* for numpy development rather than usage.



Use suppress for readability
-------------------------------

::

    In [15]: a
    Out[15]: 
    array([[[3.0e+02, 3.0e+02, 2.0e+02, 0.0e+00],
            [0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
            [0.0e+00, 0.0e+00, 0.0e+00, 2.4e-44],
            [0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00]]], dtype=float32)

    In [16]: np.set_printoptions(suppress=True)

    In [17]: a
    Out[17]: 
    array([[[300., 300., 200.,   0.],
            [  0.,   0.,   0.,   0.],
            [  0.,   0.,   0.,   0.],
            [  0.,   0.,   0.,   0.]]], dtype=float32)



N source py2.5.1
------------------

numpy 1.8.0 requires py26+ issue::

    Reading http://pypi.python.org/simple/numpy/
    Best match: numpy 1.8.0
    Downloading https://pypi.python.org/packages/source/n/numpy/numpy-1.8.0.zip#md5=6c918bb91c0cfa055b16b13850cfcd6e
    ...
      File "/data1/env/system/python/Python-2.5.1/lib/python2.5/site-packages/setuptools-0.6c9-py2.5.egg/setuptools/sandbox.py", line 63, in run
      File "/data1/env/system/python/Python-2.5.1/lib/python2.5/site-packages/setuptools-0.6c9-py2.5.egg/setuptools/sandbox.py", line 29, in <lambda>
      File "setup.py", line 16
        from __future__ import division, print_function
    SyntaxError: future feature print_function is not defined

numpy 1.7.1::

    python- source 
    np-get 
    np-build
    np-install

    # maybe harmless:

    changing mode of build/scripts.linux-i686-2.5/f2py from 664 to 775
    Exception exceptions.AttributeError: "'NoneType' object has no attribute 'maxint'" in <bound method Popen.__del__ of <subprocess.Popen object at 0xb7d7fd0c>> ignored




EOU
}
np-dir(){ echo $(local-base)/env/python/$(np-nam) ; }
np-cd(){  cd $(np-dir); }
np-mate(){ mate $(np-dir) ; }

np-ver(){ echo 1.7.1 ; }
np-nam(){ echo numpy-$(np-ver) ; }
np-url(){ echo "http://downloads.sourceforge.net/project/numpy/NumPy/$(np-ver)/$(np-nam).tar.gz" ; }

np-get(){
   local dir=$(dirname $(np-dir)) &&  mkdir -p $dir && cd $dir
   local url=$(np-url)
   local tgz=$(basename $url)
   local nam=${tgz/.tar.gz}
   [ ! -f "$tgz" ] && curl -L -O $url
   [ ! -d "$nam" ] && tar zxvf $tgz      

}

np-build(){
   np-cd
   which python
   python setup.py build
}

np-install(){
   np-cd
   which python
   python setup.py install
}

np-check(){
   which python
   python -c "import numpy "
}


