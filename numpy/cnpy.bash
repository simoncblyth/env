# === func-gen- : numpy/cnpy fgp numpy/cnpy.bash fgn cnpy fgh numpy
cnpy-src(){      echo numpy/cnpy.bash ; }
cnpy-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cnpy-src)} ; }
cnpy-vi(){       vi $(cnpy-source) ; }
cnpy-env(){      elocal- ; }
cnpy-usage(){ cat << EOU

CNPY
=====

* https://github.com/rogersce/cnpy.git



EOU
}
cnpy-dir(){ echo $(local-base)/env/numpy/cnpy ; }
cnpy-cd(){  cd $(cnpy-dir); }
cnpy-mate(){ mate $(cnpy-dir) ; }
cnpy-get(){
   local dir=$(dirname $(cnpy-dir)) &&  mkdir -p $dir && cd $dir


   [ ! -d cnpy ] && git clone https://github.com/rogersce/cnpy.git

}
