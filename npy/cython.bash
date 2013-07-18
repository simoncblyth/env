# === func-gen- : npy/cython fgp npy/cython.bash fgn cython fgh npy
cython-src(){      echo npy/cython.bash ; }
cython-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cython-src)} ; }
cython-vi(){       vi $(cython-source) ; }
cython-env(){      elocal- ; }
cython-usage(){
  cat << EOU


CYTHON OVERVIEW
================

#. Cython is Python with C data types.





CYTHON REFERENCES
~~~~~~~~~~~~~~~~~~~

* http://cython.org/

* http://docs.cython.org/

* http://conference.scipy.org/proceedings/SciPy2009/paper_1/
     
    * nice brief 11p tutorial 

* http://conference.scipy.org/proceedings/SciPy2009/paper_2/
    
    * not loading ... so google for the url and use google docs 

* http://sage.math.washington.edu/home/dagss/

* http://www.cython.org/release/Cython-0.13/

* http://www.cython.org/release/Cython-0.13/tests/run/
       
    * mother lode ... the tests are the best docs 

* http://www.mail-archive.com/sage-devel@googlegroups.com/msg25350.html

* http://trac.cython.org/cython_trac/query?status=new&status=assigned&status=reopened&order=priority&keywords=~numerics

* http://sage.math.washington.edu/home/dagss/cython-notur09/notur2009.pdf

* http://hg.sagemath.org/sage-main/file/120c07be6358/sage/databases/database.py#l1

* https://github.com/dagss/euroscipy2010/blob/master/gsl/spline.pyx

      * creates matplotlib fig in the pyx

* http://ehuss.org/mysql/api/
        
      * mysql-pyrex

* http://www.mail-archive.com/numpy-discussion@lists.sourceforge.net/msg03946.html

      * db to recarray timining 


PIP INSTALL ON C
~~~~~~~~~~~~~~~~~~

    [blyth@cms01 db]$ cython -V
    Cython version 0.13

    cython can gain a speed boost by "cdef"ing types 
    to hot variables 


MACPORTS ON G
~~~~~~~~~~~~~~~

Arghh,  tis getting python 2.5.6  (currently have 2.5.5)
fortunately this did not stomp on /opt/local/lib/python2.5/site-packages/

::

	sudo port install py25-cython
	...
	--->  Activating python25 @2.5.6_4

	To make python 2.5 the default (i.e. the version you get when you run
	'python'), please run:
		sudo port select --set python python25
        ...		
	--->  Activating py25-cython @0.15_0
	--->  Cleaning py25-cython


        port contents py25-cython
	...
        /opt/local/bin/cython-2.5
        /opt/local/lib/python2.5/site-packages/Cython-0.15-py2.5.egg-info
        /opt/local/lib/python2.5/site-packages/Cython/Build/BuildExecutable.py
	...
        /opt/local/lib/python2.5/site-packages/cython.py
	...
	/opt/local/lib/python2.5/site-packages/pyximport/
        ...
	/opt/local/share/doc/py25-cython/About.html
	... 
        /opt/local/share/doc/py25-cython/index.html
        ...



MACPORTS py26 ON G
~~~~~~~~~~~~~~~~~~~~



CYTHON C++
~~~~~~~~~~~





http://docs.cython.org/src/userguide/wrapping_CPlusPlus.html









EOU
}
cython-dir(){ echo $(local-base)/env/npy/cython ; }
cython-cd(){  cd $(cython-dir); }
cython-mate(){ mate $(cython-dir) ; }
cython-get(){
   local dir=$(dirname $(cython-dir)) &&  mkdir -p $dir && cd $dir

   pip install cython

}
