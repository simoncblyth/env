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


EOU
}
numpy-dir(){ echo $(local-base)/env/npy/numpy ; }
numpy-cd(){  cd $(numpy-dir); }
numpy-mate(){ mate $(numpy-dir) ; }
numpy-get(){
   local dir=$(dirname $(numpy-dir)) &&  mkdir -p $dir && cd $dir

   git clone git://github.com/numpy/numpy.git numpy


}
