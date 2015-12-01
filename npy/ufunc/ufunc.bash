# === func-gen- : npy/ufunc/ufunc fgp npy/ufunc/ufunc.bash fgn ufunc fgh npy/ufunc
ufunc-src(){      echo npy/uufunc/uufunc.bash ; }
ufunc-source(){   echo ${BASH_SOURCE:-$(env-home)/$(ufunc-src)} ; }
ufunc-vi(){       vi $(ufunc-source) ; }
ufunc-env(){      elocal- ; }
ufunc-usage(){ cat << EOU


UFUNC : Simplified NumPy Extension
===================================

* http://docs.scipy.org/doc/numpy/user/c-info.ufunc-tutorial.html



refs
-----

* :google:`github PyUFunc_FromFuncAndData`

* https://github.com/AaronParsons/aipy/blob/master/src/_cephes/_cephesmodule.c

  * nice avoidance of spam variables

* https://github.com/healpy/healpy/blob/master/healpy/src/_healpy_pixel_lib.cc
* https://github.com/healpy/healpy/blob/master/healpy/pixelfunc.py

  * multi lane output examples, python interface to ufunc module 


* http://cournape.github.io/davidc-scipy-2013/#52


ciexyz
------

For real world example see ciexyz-


logit
------

::

     ufunc-cd logit

     sudo python setup.py install

     # builds and installs into site-packages

::

    In [4]: sys.path.append("/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/npufunc_directory")

    In [5]: import npufunc

    In [6]: npufunc.logit(0.5)
    Out[6]: 0.0

    In [7]: a = np.linspace(0,1,5)

    In [8]: npufunc.logit(a)
    /usr/local/env/chroma_env/bin/ipython:1: RuntimeWarning: divide by zero encountered in logit
      #!/usr/local/env/chroma_env/bin/python
    Out[8]: array([  -inf, -1.099,  0.   ,  1.099,    inf])






EOU
}
ufunc-dir(){ echo $(env-home)/npy/ufunc ; }
ufunc-cd(){  cd $(ufunc-dir)/$1; }
