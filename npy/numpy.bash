# === func-gen- : npy/numpy fgp npy/numpy.bash fgn numpy fgh npy
numpy-src(){      echo npy/numpy.bash ; }
numpy-source(){   echo ${BASH_SOURCE:-$(env-home)/$(numpy-src)} ; }
numpy-vi(){       vi $(numpy-source) ; }
numpy-env(){      elocal- ; }
numpy-usage(){
  cat << EOU
     numpy-src : $(numpy-src)
     numpy-dir : $(numpy-dir)
     
        http://www.scipy.org/Cookbook


       http://github.com/numpy/numpy
       http://projects.scipy.org/numpy/report/6?asc=1&sort=modified&USER=anonymous


    == N : npy virtual py24 ==
 
         system python 2.4 has a yum installed numpy 1.1
         experiment with more recent numpy in virtual python : vip-npy 

    == C : in source py25  ==

       hostname   : cms01.phys.ntu.edu.tw
       version    : 2.0.0.dev-cfd4c05
       installdir : /data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/numpy

      see vip- for installation via pip into virtual python env 


EOU
}
numpy-dir(){ echo $(local-base)/env/npy/numpy ; }
numpy-cd(){  cd $(numpy-dir)/$1; }
numpy-mate(){ mate $(numpy-dir) ; }
numpy-get(){
   local dir=$(dirname $(numpy-dir)) &&  mkdir -p $dir && cd $dir
   git clone git://github.com/numpy/numpy.git numpy
}

numpy-pull(){
   numpy-cd
   git pull
   git show
}


numpy-version(){    local iwd=$PWD ; cd /tmp ; python -c "import numpy as np ; print np.__version__ " ;  cd $iwd ; }
numpy-installdir(){ python -c "import os, numpy as np ; print os.path.dirname(np.__file__) " ; }
numpy-include(){    python -c "import numpy as np ; print np.get_include() " ; }
numpy-info(){  cat << EOI  

    hostname   : $(hostname)
    version    : $(numpy-version)
    installdir : $(numpy-installdir)
    include    : $(numpy-include)

EOI
}

numpy-doc-preq(){
    pip install -U sphinx
}
numpy-doc(){
    numpy-cd doc
   [ "$(which sphinx-build)" == "" ] && echo $msg install sphinx first && return 
    make html
}


numpy-build(){
   numpy-cd
   which python
   python setup.py build

}

numpy-install(){
   numpy-cd
   which python

   ## using virtual python environments / or source python avoids sudo hassles  ...
   local cmd="python setup.py install"
   echo $msg $cmd
   eval $cmd

}


numpy-test(){
   local iwd=$PWD
   cd /tmp
   python -c 'import numpy; numpy.test()'
   cd $iwd 
}


