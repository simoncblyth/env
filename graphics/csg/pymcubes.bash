# === func-gen- : graphics/csg/pymcubes fgp graphics/csg/pymcubes.bash fgn pymcubes fgh graphics/csg
pymcubes-src(){      echo graphics/csg/pymcubes.bash ; }
pymcubes-source(){   echo ${BASH_SOURCE:-$(env-home)/$(pymcubes-src)} ; }
pymcubes-vi(){       vi $(pymcubes-source) ; }
pymcubes-env(){      elocal- ; }
pymcubes-usage(){ cat << EOU

PyMCubes
=========

* https://github.com/pmneila/PyMCubes


Install
--------

::

    simon:PyMCubes blyth$ python setup.py build -b /tmp/PyMCubes.build

    simon:PyMCubes blyth$ sudo python setup.py install


Usage Example
--------------

::


    In [1]: import mcubes

    In [2]: import numpy as np

    In [3]: X, Y, Z = np.mgrid[:30, :30, :30]

    In [4]: u = (X-15)**2 + (Y-15)**2 + (Z-15)**2 - 8**2

    In [5]: vertices, triangles = mcubes.marching_cubes(u, 0)

    In [6]: vertices
    Out[6]: 
    array([[  7.    ,  15.    ,  15.    ],
           [  7.    ,  15.    ,  15.    ],
           [  7.    ,  15.    ,  15.    ],
           ..., 
           [ 22.3333,  18.    ,  16.    ],
           [ 22.1333,  18.    ,  17.    ],
           [ 23.    ,  15.    ,  15.    ]])

    In [7]: vertices.shape
    Out[7]: (1182, 3)

    In [8]: triangles.shape
    Out[8]: (2360, 3)


Combining CSG distance functions ? 
--------------------------------------

* CSG booleans evaluation gives inside or outside 0/1, 
  but isosurface extraction needs a distance function ?

* Distance functions for each primitive are generally straightfor

* :google:`CSG primitive signed distance function`



Hmm : yields alota of tris
-------------------------------

* https://people.eecs.berkeley.edu/~jrs/meshs08/present/Andrews.pdf
* http://iquilezles.org/www/articles/distfunctions/distfunctions.htm



Broadcasting, ogrid, mgrid, 
---------------------------

* http://www.scipy-lectures.org/intro/numpy/numpy.html

::

   # broadcasting
   x, y = np.arange(5), np.arange(5)
   distance = np.sqrt(x ** 2 + y[:, np.newaxis] ** 2)

   # ogrid
   x, y = np.ogrid[0:5, 0:5]
   distance = np.sqrt(x ** 2 + y ** 2)

   # mgrid : directly provides matrices full of indices for cases where we can’t (or don’t want to) benefit from broadcasting

    In [55]: x, y = np.mgrid[0:5, 0:5]

    In [56]: x
    Out[56]: 
    array([[0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1],
           [2, 2, 2, 2, 2],
           [3, 3, 3, 3, 3],
           [4, 4, 4, 4, 4]])

    In [57]: y
    Out[57]: 
    array([[0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4]])

    In [58]: np.sqrt( x**2 + y**2 )
    Out[58]: 
    array([[ 0.    ,  1.    ,  2.    ,  3.    ,  4.    ],
           [ 1.    ,  1.4142,  2.2361,  3.1623,  4.1231],
           [ 2.    ,  2.2361,  2.8284,  3.6056,  4.4721],
           [ 3.    ,  3.1623,  3.6056,  4.2426,  5.    ],
           [ 4.    ,  4.1231,  4.4721,  5.    ,  5.6569]])
        




EOU
}
pymcubes-dir(){ echo $(local-base)/env/graphics/csg/PyMCubes ; }


pymcubes-cd(){  cd $(pymcubes-dir); }
pymcubes-mate(){ mate $(pymcubes-dir) ; }
pymcubes-get(){
   local dir=$(dirname $(pymcubes-dir)) &&  mkdir -p $dir && cd $dir

   [ ! -d PyMCubes ] && git clone https://github.com/pmneila/PyMCubes 

}
