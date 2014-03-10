# === func-gen- : zeromq/zeromq fgp zeromq/zeromq.bash fgn zeromq fgh zeromq
zeromq-src(){      echo zeromq/zeromq.bash ; }
zeromq-source(){   echo ${BASH_SOURCE:-$(env-home)/$(zeromq-src)} ; }
zeromq-vi(){       vi $(zeromq-source) ; }
zeromq-env(){      elocal- ; }
zeromq-usage(){ cat << EOU

ZEROMQ
======


* http://zguide.zeromq.org/py:all




EOU
}
zeromq-dir(){ echo $(local-base)/env/zeromq/zeromq-zeromq ; }
zeromq-cd(){  cd $(zeromq-dir); }
zeromq-mate(){ mate $(zeromq-dir) ; }
zeromq-get(){
   local dir=$(dirname $(zeromq-dir)) &&  mkdir -p $dir && cd $dir

}


zeromq-zguide-get(){

  git clone --depth=1 git://github.com/imatix/zguide.git
}

