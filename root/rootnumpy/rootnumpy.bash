# === func-gen- : root/rootnumpy/rootnumpy fgp root/rootnumpy/rootnumpy.bash fgn rootnumpy fgh root/rootnumpy
rootnumpy-src(){      echo root/rootnumpy/rootnumpy.bash ; }
rootnumpy-source(){   echo ${BASH_SOURCE:-$(env-home)/$(rootnumpy-src)} ; }
rootnumpy-vi(){       vi $(rootnumpy-source) ; }
rootnumpy-env(){      elocal- ; }
rootnumpy-usage(){ cat << EOU

* http://rootpy.github.io/root_numpy/install.html




EOU
}
rootnumpy-dir(){ echo $(local-base)/env/root/rootnumpy ; }
rootnumpy-cd(){  cd $(rootnumpy-dir); }
rootnumpy-get(){
   local dir=$(dirname $(rootnumpy-dir)) &&  mkdir -p $dir && cd $dir

   git clone git://github.com/rootpy/root_numpy.git
}
