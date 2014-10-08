# === func-gen- : chroma/chromacpp/chromacpp fgp chroma/chromacpp/chromacpp.bash fgn chromacpp fgh chroma/chromacpp
chromacpp-src(){      echo chroma/chromacpp/chromacpp.bash ; }
chromacpp-source(){   echo ${BASH_SOURCE:-$(env-home)/$(chromacpp-src)} ; }
chromacpp-vi(){       vi $(chromacpp-source) ; }
chromacpp-env(){      elocal- ; }
chromacpp-usage(){ cat << EOU





EOU
}
chromacpp-dir(){ echo $(env-home)/chroma/chromacpp ; }
chromacpp-cd(){  cd $(chromacpp-dir); }
chromacpp-mate(){ mate $(chromacpp-dir) ; }
chromacpp-get(){
   local dir=$(dirname $(chromacpp-dir)) &&  mkdir -p $dir && cd $dir

}


chromacpp-build(){

   chromacpp-cd

   clang chromacpp.c -g -c
   clang npyreader.c -g -c

   clang *.o -o chromacpp

}

