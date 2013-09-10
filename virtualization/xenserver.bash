# === func-gen- : virtualization/xenserver fgp virtualization/xenserver.bash fgn xenserver fgh virtualization
xenserver-src(){      echo virtualization/xenserver.bash ; }
xenserver-source(){   echo ${BASH_SOURCE:-$(env-home)/$(xenserver-src)} ; }
xenserver-vi(){       vi $(xenserver-source) ; }
xenserver-env(){      elocal- ; }
xenserver-usage(){ cat << EOU

XENSERVER
==========

* http://www.xenserver.org/
* http://www.xenserver.org/overview-xenserver-open-source-virtualization/download.html



EOU
}
xenserver-dir(){ echo $(local-base)/env/virtualization/virtualization-xenserver ; }
xenserver-cd(){  cd $(xenserver-dir); }
xenserver-mate(){ mate $(xenserver-dir) ; }
xenserver-get(){
   local dir=$(dirname $(xenserver-dir)) &&  mkdir -p $dir && cd $dir

}
