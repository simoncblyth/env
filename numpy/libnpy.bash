# === func-gen- : numpy/libnpy fgp numpy/libnpy.bash fgn libnpy fgh numpy
libnpy-src(){      echo numpy/libnpy.bash ; }
libnpy-source(){   echo ${BASH_SOURCE:-$(env-home)/$(libnpy-src)} ; }
libnpy-vi(){       vi $(libnpy-source) ; }
libnpy-env(){      elocal- ; }
libnpy-usage(){ cat << EOU

LIBNPY
======

* http://wiki.scipy.org/Cookbook/InputOutput



EOU
}


libnpy-name(){ echo libnpy-0.5 ; }
libnpy-dir(){ echo $(local-base)/env/numpy/$(libnpy-name) ; }
libnpy-cd(){  cd $(libnpy-dir); }
libnpy-mate(){ mate $(libnpy-dir) ; }
libnpy-get(){
   local dir=$(dirname $(libnpy-dir)) &&  mkdir -p $dir && cd $dir
   local nam=$(libnpy-name)

   [ ! -f "$nam.tgz" ] && curl -L -O $(libnpy-url) 
   [ ! -d "$nam" ] && tar zxvf $nam.tgz
}

libnpy-url(){
    echo http://www.maths.unsw.edu.au/~mclean/libnpy-0.5.tgz 
}

