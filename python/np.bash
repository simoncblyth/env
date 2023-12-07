# === func-gen- : python/np fgp python/np.bash fgn np fgh python
np-src(){      echo python/np.bash ; }
np-source(){   echo ${BASH_SOURCE:-$(env-home)/$(np-src)} ; }
np-vi(){       vi $(np-source) ; }
np-env(){      elocal- ; }
np-usage(){ cat << EOU

Plain Vanilla Usage of numpy
==============================

See also *numpy-* for numpy development rather than usage.



np.char.startswith
------------------


::

    In [10]: np.flatnonzero(np.char.startswith(b.q, "TO BT SD".encode("utf-8") ))
    Out[10]: array([   4,    7,   18,   21,   22, ..., 9990, 9993, 9994, 9997, 9998])



np.newaxis example
--------------------

Three timestamps per event::

    In [30]: ap[:,:,0]
    Out[30]: 
    array([[1700396795214975, 1700396795215091, 1700396795248809],
           [1700396797107438, 1700396797107538, 1700396797122071],
           [1700396798401574, 1700396798401679, 1700396798416074],
           [1700396799679454, 1700396799679554, 1700396799694040],
           [1700396800979182, 1700396800979289, 1700396800994112],
           [1700396802257773, 1700396802257876, 1700396802272298],
           [1700396803527965, 1700396803528071, 1700396803542506],
           [1700396804821025, 1700396804821126, 1700396804835753],
           [1700396806108956, 1700396806109061, 1700396806123453],
           [1700396807386630, 1700396807386737, 1700396807401141]])

    In [41]: ap[:,:,0] - ap[:,:,0].min()       ## subtract the earliest timestamp from all 
    Out[41]:
    array([[       0,      116,    33834],
           [ 1892463,  1892563,  1907096],
           [ 3186599,  3186704,  3201099],
           [ 4464479,  4464579,  4479065],
           [ 5764207,  5764314,  5779137],
           [ 7042798,  7042901,  7057323],
           [ 8312990,  8313096,  8327531],
           [ 9606050,  9606151,  9620778],
           [10893981, 10894086, 10908478],
           [12171655, 12171762, 12186166]], dtype=int64)

    In [37]: ap[:,:,0] - ap[:,0,0,np.newaxis]     ## subtract the first column from each, to see BoE relative times
    Out[37]:
    array([[    0,   116, 33834],
           [    0,   100, 14633],
           [    0,   105, 14500],
           [    0,   100, 14586],
           [    0,   107, 14930],
           [    0,   103, 14525],
           [    0,   106, 14541],
           [    0,   101, 14728],
           [    0,   105, 14497],
           [    0,   107, 14511]])

    In [38]: ap[:,:,0] - ap[:,1,0,np.newaxis]   ## subtract mid column 
    Out[38]:
    array([[ -116,     0, 33718],
           [ -100,     0, 14533],
           [ -105,     0, 14395],
           [ -100,     0, 14486],
           [ -107,     0, 14823],
           [ -103,     0, 14422],
           [ -106,     0, 14435],
           [ -101,     0, 14627],
           [ -105,     0, 14392],
           [ -107,     0, 14404]])

    In [39]: ap[:,:,0] - ap[:,2,0,np.newaxis]    ## subtract last column 
    Out[39]:
    array([[-33834, -33718,      0],
           [-14633, -14533,      0],
           [-14500, -14395,      0],
           [-14586, -14486,      0],
           [-14930, -14823,      0],
           [-14525, -14422,      0],
           [-14541, -14435,      0],
           [-14728, -14627,      0],
           [-14497, -14392,      0],
           [-14511, -14404,      0]])


Another example of just handling 1st column::

    In [41]: a[:,:,0] - a[:,:,0][:,0,np.newaxis]
    Out[41]: 
    array([[      0,     131,  459159],
           [      0,     184,  791004],
           [      0,     175, 1195828]])



np.newaxis for subtracting off first column value for every row of 2D array
-----------------------------------------------------------------------------

::

    pp = a.pf[a.pfr>20,1:][10:]  

    In [23]: pp.shape
    Out[23]: (3783, 11)

    In [24]: pp[:,0].shape
    Out[24]: (3783,)

    In [25]: pp[:,0,np.newaxis].shape   ## adding an axis allows element-by-element subtraction
    Out[25]: (3783, 1)

    In [26]: pp[:,0,None].shape
    Out[26]: (3783, 1)

    In [22]: pp - pp[:,0,None]
    Out[22]:#0   1   2   3   4-->5
    array([[ 0,  1,  1,  1,  1, 51, 52, 53, 53, 53, 57],
           [ 0,  1,  1,  1,  1, 51, 51, 53, 53, 53, 56],
           [ 0,  0,  1,  1,  1, 51, 52, 53, 53, 53, 56],
           [ 0,  1,  1,  1,  1, 52, 52, 53, 53, 54, 57],
           [ 0,  0,  0,  0,  0, 50, 50, 52, 52, 52, 54],
           [ 0,  0,  0,  0,  0, 50, 50, 51, 51, 52, 54],
           [ 0,  0,  1,  1,  1, 51, 52, 53, 53, 53, 57],
           [ 0,  0,  1,  1,  1, 53, 53, 55, 55, 55, 61],
           [ 0,  0,  0,  0,  0, 50, 51, 52, 52, 52, 55],
           [ 0,  0,  0,  0,  0, 51, 51, 52, 52, 53, 56],
           ...,
           [ 0,  0,  1,  1,  1, 51, 51, 52, 52, 52, 55],
           [ 0,  0,  1,  1,  1, 52, 52, 53, 54, 54, 57],
           [ 0,  0,  1,  1,  1, 51, 51, 53, 53, 53, 58],
           [ 0,  1,  1,  1,  1, 52, 52, 54, 54, 54, 58],
           [ 0,  0,  0,  1,  1, 51, 51, 53, 53, 53, 56],
           [ 0,  0,  1,  1,  1, 50, 51, 52, 52, 52, 56],
           [ 0,  0,  0,  0,  0, 52, 52, 54, 54, 54, 58],
           [ 0,  0,  1,  1,  1, 51, 51, 53, 53, 53, 56],
           [ 0,  0,  0,  0,  0, 51, 51, 53, 53, 53, 57],
           [ 0,  0,  1,  1,  1, 52, 52, 53, 53, 54, 57]], dtype=uint64)





Ring buffer : with no.take and wrap mode
------------------------------------------

https://stackoverflow.com/questions/28398220/circular-numpy-array-indices

::

    In [21]: a = np.arange(30, dtype=np.float32).reshape(-1,3) ; a
    Out[21]: 
    array([[ 0.,  1.,  2.],
           [ 3.,  4.,  5.],
           [ 6.,  7.,  8.],
           [ 9., 10., 11.],
           [12., 13., 14.],
           [15., 16., 17.],
           [18., 19., 20.],
           [21., 22., 23.],
           [24., 25., 26.],
           [27., 28., 29.]], dtype=float32)

    In [22]: np.take( a, np.arange(15), mode='wrap', axis=0 )
    Out[22]: 
    array([[ 0.,  1.,  2.],
           [ 3.,  4.,  5.],
           [ 6.,  7.,  8.],
           [ 9., 10., 11.],
           [12., 13., 14.],
           [15., 16., 17.],
           [18., 19., 20.],
           [21., 22., 23.],
           [24., 25., 26.],
           [27., 28., 29.],
           [ 0.,  1.,  2.],
           [ 3.,  4.,  5.],
           [ 6.,  7.,  8.],
           [ 9., 10., 11.],
           [12., 13., 14.]], dtype=float32)





HEP starting to see the NumPy light
-------------------------------------

* https://github.com/scikit-hep/awkward-array#relationship-to-arrow
* https://docs.google.com/document/d/1lj8ARTKV1_hqGTh0W_f01S6SsmpzZAXz9qqqWnEB3j4/edit#heading=h.ymw0o054nurw

* https://iris-hep.org/projects/awkward.html


Compressed NumPy ? bcolz
---------------------------

* http://bcolz.blosc.org/en/latest/intro.html
* https://github.com/Blosc/bcolz/blob/master/docs/tutorial_carray.ipynb

* blosc : faster than *memcpy()*


Faster NumPy ? numexpr
-------------------------

* https://github.com/pydata/numexpr
* https://numexpr.readthedocs.io/en/latest/user_guide.html


Beyond NumPy : many packages provide different implementation of the NumPy API 
-------------------------------------------------------------------------------

* https://matthewrocklin.com/blog//work/2018/05/27/beyond-numpy


Best Introductions
--------------------

* https://webvalley.fbk.eu/static/media/uploads/presentations/introductiontonumpy2.pdf
* ~/opticks_refs/numpy/introductiontonumpy2.pdf 


* http://csc.ucdavis.edu/~chaos/courses/nlp/Software/NumPyBook.pdf
* ~/opticks_refs/numpy/Oliphant_NumPyBook.pdf 


Intros
--------

* http://acme.byu.edu/wp-content/uploads/2017/08/NumpyIntro.pdf


* https://sebastianraschka.com/pdf/books/dlb/appendix_f_numpy-intro.pdf
* ~/opticks_refs/numpy/sebastianraschka_appendix_f_numpy-intro.pdf 

While adding and removing elements from the end of a Python list is very
efficient, altering the size of a NumPy array is very expensive since it
requires creating a new array and carrying over the contents of the old


* need to understand how numpy is implemented to use it effectively 


Why NumPy 
----------

* https://www.stat.washington.edu/~hoytak/blog/whypython.html


Zen of NumPy
-------------

* https://github.com/numpy/numpy/issues/2389

The Zen of Numpy, by Travis Oliphant  (version 0.1)
Strided is better than scattered.
Contiguous is better than strided.
Descriptive is better than imperative (e.g. data-types).
Array-orientated is better than object-oriented.
Broadcasting is a great idea -- use where possible!
Vectorized is better than an explicit loop.
Unless it's complicated -- then use Cython or numexpr.
Think in higher dimensions.







Also ran Introductions
-----------------------

* https://engineering.ucsb.edu/~shell/che210d/numpy.pdf
* http://www.datadependence.com/2016/05/scientific-python-numpy/


Random reseed
---------------

::

   np.random.seed(10)
   a = np.random.random_sample( (1000000,4,4) )
   np.random.seed(10)
   b = np.random.random_sample( (1000000,4,4) )

   np.save("a.npy", a )
   np.save("b.npy", b )

   a[0]
   b[0]
   a - b
   np.max( np.abs(a - b ))



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


