# === func-gen- : messaging/gason fgp messaging/gason.bash fgn gason fgh messaging
gason-src(){      echo messaging/gason.bash ; }
gason-source(){   echo ${BASH_SOURCE:-$(env-home)/$(gason-src)} ; }
gason-vi(){       vi $(gason-source) ; }
gason-env(){      elocal- ; }
gason-usage(){ cat << EOU

GASON
======

https://github.com/vivkin/gason

* disqualified by use of C++11 : as presume that would cause portability problems 


EOU
}
gason-dir(){ echo $(local-base)/env/messaging/gason ; }
gason-cd(){  cd $(gason-dir); }
gason-mate(){ mate $(gason-dir) ; }
gason-get(){
   local dir=$(dirname $(gason-dir)) &&  mkdir -p $dir && cd $dir

   git clone https://github.com/vivkin/gason.git 


}
