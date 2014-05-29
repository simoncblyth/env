# === func-gen- : tools/ld fgp tools/ld.bash fgn ld fgh tools
ld-src(){      echo tools/ld.bash ; }
ld-source(){   echo ${BASH_SOURCE:-$(env-home)/$(ld-src)} ; }
ld-vi(){       vi $(ld-source) ; }
ld-env(){      elocal- ; }
ld-usage(){ cat << EOU

LD DEBUGGING
=============

Dump library loads and missing symbols::

    LD_DEBUG=libs python somescript.py 




EOU
}
ld-dir(){ echo $(local-base)/env/tools/tools-ld ; }
ld-cd(){  cd $(ld-dir); }
ld-mate(){ mate $(ld-dir) ; }
ld-get(){
   local dir=$(dirname $(ld-dir)) &&  mkdir -p $dir && cd $dir

}
