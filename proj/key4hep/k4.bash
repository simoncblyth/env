# === func-gen- : proj/key4hep/k4 fgp proj/key4hep/k4.bash fgn k4 fgh proj/key4hep src base/func.bash
k4-source(){   echo ${BASH_SOURCE} ; }
k4-edir(){ echo $(dirname $(k4-source)) ; }
k4-ecd(){  cd $(k4-edir); }
k4-dir(){  echo $LOCAL_BASE/env/proj/key4hep/k4 ; }
k4-cd(){   cd $(k4-dir); }
k4-vi(){   vi $(k4-source) ; }
k4-env(){  elocal- ; }
k4-usage(){ cat << EOU


Key4HEP: Turnkey Software for Future Colliders 

https://github.com/key4hep


https://indico.jlab.org/event/420/contributions/8308/attachments/6909/9428/210504_sailer_key4hep.pdf


EOU
}
k4-get(){
   local dir=$(dirname $(k4-dir)) &&  mkdir -p $dir && cd $dir

}
