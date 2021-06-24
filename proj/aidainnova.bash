# === func-gen- : proj/aidainnova fgp proj/aidainnova.bash fgn aidainnova fgh proj src base/func.bash
aidainnova-source(){   echo ${BASH_SOURCE} ; }
aidainnova-edir(){ echo $(dirname $(aidainnova-source)) ; }
aidainnova-ecd(){  cd $(aidainnova-edir); }
aidainnova-dir(){  echo $LOCAL_BASE/env/proj/aidainnova ; }
aidainnova-cd(){   cd $(aidainnova-dir); }
aidainnova-vi(){   vi $(aidainnova-source) ; }
aidainnova-env(){  elocal- ; }
aidainnova-usage(){ cat << EOU

AIDA : Advancement and Innovation for Detectors at Accelerators

* https://indico.cern.ch/event/1028932/contributions/4320412/attachments/2228331/3775233/AIDAinnova-SFT-2021-04-19.pdf


EOU
}
aidainnova-get(){
   local dir=$(dirname $(aidainnova-dir)) &&  mkdir -p $dir && cd $dir

}
