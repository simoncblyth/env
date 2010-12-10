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


    == debug build ==

       http://projects.scipy.org/numpy/ticket/539

       numpy-cd ; rm -rf build ; numpy-build --debug 


    == N : npy virtual py24 ==
 
         system python 2.4 has a yum installed numpy 1.1
         experiment with more recent numpy in virtual python : vip-npy 

    == C : in source py25  ==

       hostname   : cms01.phys.ntu.edu.tw
       version    : 2.0.0.dev-cfd4c05
       installdir : /data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/numpy

      see vip- for installation via pip into virtual python env 

    == G : (macports) python_select python27 ==

      Numpy installs into :
        /opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages/numpy/


   == Fork numpy on github  ==

     Follow http://help.github.com/forking/
     clone the fork with 
   
         git clone git@github.com:scb-/numpy.git
         cd numpy
         git remote add upstream git://github.com/numpy/numpy.git
         git fetch upstream

    Thence can make changes and locally commut changes to my numpy fork 
    and push to (my) master on github with :

         git push origin master

    After which can issue   http://help.github.com/pull-requests/




   == Unsure if still need for this change .. ==

   Was triggering segv in doing repr of arrays derived from buffers ...
   need to capture issue in test 

{{{
[blyth@cms01 tests]$ git diff
diff --git a/numpy/core/src/multiarray/scalarapi.c b/numpy/core/src/multiarray/scalarapi.c
index 87e140c..0f84d87 100644
--- a/numpy/core/src/multiarray/scalarapi.c
+++ b/numpy/core/src/multiarray/scalarapi.c
@@ -674,7 +674,7 @@ PyArray_Scalar(void *data, PyArray_Descr *descr, PyObject *base)
         memcpy(&(((PyDatetimeScalarObject *)obj)->obmeta), dt_data,
                sizeof(PyArray_DatetimeMetaData));
     }
-    if (PyTypeNum_ISFLEXIBLE(type_num)) {
+    if (PyTypeNum_ISEXTENDED(type_num)) {
         if (type_num == PyArray_STRING) {
             destptr = PyString_AS_STRING(obj);
             ((PyStringObject *)obj)->ob_shash = -1;

}}}



EOU
}
numpy-dir(){ echo $(local-base)/env/npy/numpy ; }
numpy-cd(){  cd $(numpy-dir)/$1; }
numpy-scd(){  cd $(env-home)/npy/numpy/$1; }
numpy-mate(){ mate $(numpy-dir) ; }

#numpy-name(){ echo upstream ; }
numpy-name(){ 
  case $USER in 
    blyth) echo scbforkrw ;; 
        *) echo scbforkro ;; 
  esac
}

numpy-url(){
   case $(numpy-name) in 
      upstream) echo git://github.com/numpy/numpy.git ;;
     scbforkrw) echo git@github.com:scb-/numpy.git ;;
     scbforkro) echo git://github.com:scb-/numpy.git ;;
   esac
}

numpy-get(){
   local msg="=== $FUNCNAME :"
   local dir=$(dirname $(numpy-dir)) &&  mkdir -p $dir && cd $dir
   [ -d numpy ] && echo numpy exists already, remove before re-cloneing  && return 1 
   local cmd="git clone $(numpy-url) numpy"
   echo $msg $cmd
   eval $cmd
}
numpy-wipe(){
   local dir=$(dirname $(numpy-dir)) &&  mkdir -p $dir && cd $dir
   rm -rf numpy
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


