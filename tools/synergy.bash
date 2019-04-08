# === func-gen- : tools/synergy fgp tools/synergy.bash fgn synergy fgh tools src base/func.bash
synergy-source(){   echo ${BASH_SOURCE} ; }
synergy-edir(){ echo $(dirname $(synergy-source)) ; }
synergy-ecd(){  cd $(synergy-edir); }
synergy-dir(){  echo $LOCAL_BASE/env/tools/synergy ; }
synergy-cd(){   cd $(synergy-dir); }
synergy-vi(){   vi $(synergy-source) ; }
synergy-env(){  elocal- ; }
synergy-usage(){ cat << EOU


Synergy
========

Share one mouse, one keyboard, and one clipboard between multiple Windows, Mac,
and Linux computers.

* https://alternativeto.net/software/synergy/
* https://alternative.me/synergy


* https://github.com/debauchee/barrier forked from https://github.com/symless/synergy-core



EOU
}
synergy-get(){
   local dir=$(dirname $(synergy-dir)) &&  mkdir -p $dir && cd $dir

}
