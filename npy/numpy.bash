numpy-source(){   echo ${BASH_SOURCE} ; }
numpy-vi(){       vi $(numpy-source) ; }
numpy-env(){      elocal- ; }
numpy-usage(){  cat << EOU

For Usage tips see np-


* https://github.com/xtensor-stack/xtensor
* https://towardsdatascience.com/the-xtensor-vision-552dd978e9ad
* https://xtensor.readthedocs.io/en/latest/


EOU
}

numpy-dir(){ 
   ## virtualized takes precedent 
   [ -n "$VIRTUAL_ENV" ] && echo $VIRTUAL_ENV/src/numpy && return 
   echo $(local-base)/env/npy/$(numpy-name) ; 
}
numpy-cd(){  cd $(numpy-dir)/$1; }
numpy-scd(){  cd $(env-home)/npy/numpy/$1; }

numpy-name(){ 
    case $USER in
      blyth) echo scbfork ;;
      thho) echo scbfork_ro ;; 
    esac
}

numpy-url(){
   case $(numpy-name) in 
      upstream) echo git://github.com/numpy/numpy.git ;;
       scbfork) echo git@github.com:scb-/numpy.git    ;;
    scbfork_ro) echo git://github.com/scb-/numpy.git  ;;
   esac
}


numpy-tute(){
   cd $(local-base)/env/npy/ ; 

   local name=Euroscipy-intro-tutorials
   [ ! -d "$name" ] &&  git clone https://github.com/scipy-lectures/$name.git

   cd $name/lecture_notes
   make html
}
   ## hmmm needs : scipy.interpolate

# git clone https://github.com/numpy/numpy.git numpy

numpy-get(){
   local msg="=== $FUNCNAME :"
   local dir=$(dirname $(numpy-dir)) &&  mkdir -p $dir && cd $dir
   [ -d numpy ] && echo numpy exists already, remove before re-cloneing  && return 1 
   git clone $(numpy-url) $(numpy-name)
   numpy-cd
   git branch 
}
numpy-wipe(){
   local dir=$(dirname $(numpy-dir)) &&  mkdir -p $dir && cd $dir
   rm -rf numpy
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
numpy-nginx(){
   nginx-
   local docd=$(numpy-dir)/doc/build/html/ 
   [ ! -d "$docd" ] && echo $msg ERROR no docs dir at  $docd && return
   nginx-ln $docd numpy
}



numpy-build(){
   numpy-cd
   which python
   python setup.py build $*
}

numpy-install(){
   numpy-cd
   which python
   ## using virtual python environments / or source python avoids sudo hassles  ...
   local cmd="python setup.py install"
   echo $msg $cmd
   eval $cmd
}

numpy-sinstall(){
   numpy-cd
   which python
   ## using virtual python environments / or source python avoids sudo hassles  ...
   local cmd="sudo python setup.py install"
   echo $msg $cmd
   eval $cmd
}


numpy-test(){
   local iwd=$PWD
   cd /tmp
   python -c 'import numpy; numpy.test()'
   cd $iwd 
}


