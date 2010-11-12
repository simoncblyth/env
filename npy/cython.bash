# === func-gen- : npy/cython fgp npy/cython.bash fgn cython fgh npy
cython-src(){      echo npy/cython.bash ; }
cython-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cython-src)} ; }
cython-vi(){       vi $(cython-source) ; }
cython-env(){      elocal- ; }
cython-usage(){
  cat << EOU
     cython-src : $(cython-src)
     cython-dir : $(cython-dir)

     http://cython.org/
     http://docs.cython.org/


     http://conference.scipy.org/proceedings/SciPy2009/paper_1/
          nice brief 11p tutorial 

     http://conference.scipy.org/proceedings/SciPy2009/paper_2/
          not loading ... so google for the url and use google docs 


    C    pip install cython   : installed Cython-0.13.tar.gz

[blyth@cms01 db]$ cython -V
Cython version 0.13


    cython can gain a speed boost by "cdef"ing types 
    to hot variables 


EOU
}
cython-dir(){ echo $(local-base)/env/npy/cython ; }
cython-cd(){  cd $(cython-dir); }
cython-mate(){ mate $(cython-dir) ; }
cython-get(){
   local dir=$(dirname $(cython-dir)) &&  mkdir -p $dir && cd $dir

   pip install cython

}
