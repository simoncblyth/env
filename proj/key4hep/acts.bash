# === func-gen- : proj/key4hep/acts fgp proj/key4hep/acts.bash fgn acts fgh proj/key4hep src base/func.bash
acts-source(){   echo ${BASH_SOURCE} ; }
acts-edir(){ echo $(dirname $(acts-source)) ; }
acts-ecd(){  cd $(acts-edir); }
acts-dir(){  echo $LOCAL_BASE/env/proj/key4hep/acts ; }
acts-cd(){   cd $(acts-dir); }
acts-vi(){   vi $(acts-source) ; }
acts-env(){  elocal- ; }
acts-usage(){ cat << EOU


ACTS: from ATLAS software towards a common track reconstruction software
https://cds.cern.ch/record/2243297?ln=en


https://acts-project.github.io
https://gitlab.cern.ch/acts


https://github.com/acts-project/acts

Experiment-independent toolkit for (charged) particle track reconstruction in
(high energy) physics experiments implemented in modern C++

https://github.com/acts-project/detray
Test library for detector surface intersection

https://github.com/acts-project/traccc
Demonstrator tracking chain on accelerators




EOU
}
acts-get(){
   local dir=$(dirname $(acts-dir)) &&  mkdir -p $dir && cd $dir

}
