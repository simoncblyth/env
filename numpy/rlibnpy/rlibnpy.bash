# === func-gen- : numpy/rlibnpy fgp numpy/rlibnpy.bash fgn rlibnpy fgh numpy
rlibnpy-src(){      echo numpy/rlibnpy/rlibnpy.bash ; }
rlibnpy-source(){   echo ${BASH_SOURCE:-$(env-home)/$(rlibnpy-src)} ; }
rlibnpy-vi(){       vi $(rlibnpy-source) ; }
rlibnpy-env(){      elocal- ; }
rlibnpy-usage(){ cat << EOU

Reimplementation of libnpy as single C++ header
=================================================

Original source. 

   git clone https://gist.github.com/5656056.git rlibnpy 

Modified to allow use with memory buffers in addition 
to files.




EOU
}
rlibnpy-dir(){ echo $(local-base)/env/numpy/rlibnpy ; }
rlibnpy-cd(){  cd $(rlibnpy-dir); }
rlibnpy-mate(){ mate $(rlibnpy-dir) ; }
rlibnpy-get(){
   local dir=$(dirname $(rlibnpy-dir)) &&  mkdir -p $dir && cd $dir



}
