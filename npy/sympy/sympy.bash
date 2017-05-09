# === func-gen- : npy/sympy/sympy fgp npy/sympy/sympy.bash fgn sympy fgh npy/sympy
sympy-src(){      echo npy/sympy/sympy.bash ; }
sympy-source(){   echo ${BASH_SOURCE:-$(env-home)/$(sympy-src)} ; }
sympy-vi(){       vi $(sympy-source) ; }
sympy-env(){      elocal- ; }
sympy-usage(){ cat << EOU

SymPy : library for symbolic mathematics
===========================================

* https://github.com/sympy/sympy
* http://docs.sympy.org/latest/index.html 
* http://docs.sympy.org/latest/tutorial/index.html#tutorial
* http://docs.sympy.org/latest/tutorial/manipulation.html

  SymPy Expression trees : perhaps can simplify deep CSG trees using this ?

Tutorial pages feature a cute "SymPy Live Shell" panel in browser

::

    >>> x = symbols('x')
    >>> expr = x + 1
    >>> x = 2            # changes python variable, not the sympy symbol
    >>> print(expr)
    x + 1



::

    >>> from sympy import Symbol, cos
    >>> x = Symbol('x')
    >>> e = 1/cos(x)
    >>> print e.series(x, 0, 10)
    1 + x**2/2 + 5*x**4/24 + 61*x**6/720 + 277*x**8/8064 + O(x**10)



mpmath dependency 
-------------------

* http://mpmath.org

mpmath is a free (BSD licensed) Python library for real and complex
floating-point arithmetic with arbitrary precision. It has been developed by
Fredrik Johansson since 2007, with help from many contributors.




EOU
}
sympy-dir(){ echo $(local-base)/env/npy/sympy/npy/sympy-sympy ; }
sympy-cd(){  cd $(sympy-dir); }
sympy-mate(){ mate $(sympy-dir) ; }
sympy-get(){
   local dir=$(dirname $(sympy-dir)) &&  mkdir -p $dir && cd $dir

}
