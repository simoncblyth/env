# === func-gen- : npy/numpy fgp npy/numpy.bash fgn numpy fgh npy
numpy-src(){      echo npy/numpy.bash ; }
numpy-source(){   echo ${BASH_SOURCE:-$(env-home)/$(numpy-src)} ; }
numpy-vi(){       vi $(numpy-source) ; }
numpy-env(){      elocal- ; }
numpy-usage(){
  cat << EOU
     numpy-src : $(numpy-src)
     numpy-dir : $(numpy-dir)


   numpy in use on C came as dependency of matplotlib (Oct 2010)
       In [2]: np.__version__
       Out[2]: '1.5.0'
       In [3]: np.__file__
       Out[3]: '/data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/numpy/__init__.pyc'    


    On N ...
         system python 2.4 has a yum installed numpy 1.1
         experiment with more recent numpy in virtual python : vip-npy 


      see vip- for installation via pip into virtual python env 


     http://www.scipy.org/Cookbook


EOU
}
numpy-dir(){ echo $(local-base)/env/npy/numpy ; }
numpy-cd(){  cd $(numpy-dir); }
numpy-mate(){ mate $(numpy-dir) ; }
numpy-get(){
   local dir=$(dirname $(numpy-dir)) &&  mkdir -p $dir && cd $dir

   git clone git://github.com/numpy/numpy.git numpy
}


numpy-build(){
   numpy-cd
   which python
   python setup.py build

}

numpy-install(){
   numpy-cd
   which python


   ## using virtual python environments ... avoids sudo hassles  ... ge

   local cmd="python setup.py install"
   echo $msg $cmd
   eval $cmd

}

