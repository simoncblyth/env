# === func-gen- : npy/scipy fgp npy/scipy.bash fgn scipy fgh npy
scipy-src(){      echo npy/scipy.bash ; }
scipy-source(){   echo ${BASH_SOURCE:-$(env-home)/$(scipy-src)} ; }
scipy-vi(){       vi $(scipy-source) ; }
scipy-env(){      elocal- ; }
scipy-usage(){  cat << EOU

scipy
========

* http://www.scipy.org
* https://docs.scipy.org/doc/scipy/reference/tutorial/index.html


installs
----------

epsilon
    using conda base env with python3



optimize
----------

* https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html


* https://en.wikipedia.org/wiki/Rosenbrock_function

::

    In [7]: import numpy as np

    In [5]: rosen = lambda x:sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

    In [8]: x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])

    In [9]: rosen(x0)
    Out[9]: 848.22

    In [10]: res = minimize(rosen, x0, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 339
             Function evaluations: 571

    In [11]: res
    Out[11]: 
     final_simplex: (array([
           [1.        , 1.        , 1.        , 1.        , 1.        ],
           [1.        , 1.        , 1.        , 1.        , 1.        ],
           [1.        , 1.        , 1.        , 1.00000001, 1.00000001],
           [1.        , 1.        , 1.        , 1.        , 1.        ],
           [1.        , 1.        , 1.        , 1.        , 1.        ],
           [1.        , 1.        , 1.        , 1.        , 0.99999999]]), 
          array([4.86115343e-17, 7.65182843e-17, 8.11395684e-17, 8.63263255e-17,8.64080682e-17, 2.17927418e-16]))
               fun: 4.861153433422115e-17
           message: 'Optimization terminated successfully.'
              nfev: 571
               nit: 339
            status: 0
           success: True
                 x: array([1., 1., 1., 1., 1.])





EOU
}
scipy-dir(){ echo $(local-base)/env/npy/npy-scipy ; }
scipy-cd(){  cd $(scipy-dir); }
