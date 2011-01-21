# === func-gen- : npy/scipy fgp npy/scipy.bash fgn scipy fgh npy
scipy-src(){      echo npy/scipy.bash ; }
scipy-source(){   echo ${BASH_SOURCE:-$(env-home)/$(scipy-src)} ; }
scipy-vi(){       vi $(scipy-source) ; }
scipy-env(){      elocal- ; }
scipy-usage(){
  cat << EOU
     scipy-src : $(scipy-src)
     scipy-dir : $(scipy-dir)

     http://www.scipy.org
     http://scipy.org/Installing_SciPy/BuildingGeneral

     N ... failing to build for lack of BLAS ... 



EOU
}
scipy-dir(){ echo $(local-base)/env/npy/npy-scipy ; }
scipy-cd(){  cd $(scipy-dir); }
scipy-mate(){ mate $(scipy-dir) ; }
scipy-get(){
   local msg="=== $FUNCNAME :"
   local dir=$(dirname $(scipy-dir)) &&  mkdir -p $dir && cd $dir
   [ -z "$VIRTUAL_ENV" ]  && echo $msg get virtual OR get lost && return 1 

    pip install -E $VIRTUAL_ENV -e svn+http://svn.scipy.org/svn/scipy/trunk#egg=scipy

}
