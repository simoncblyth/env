# === func-gen- : graphics/atb/atb fgp graphics/atb/atb.bash fgn atb fgh graphics/atb
atb-src(){      echo graphics/atb/atb.bash ; }
atb-source(){   echo ${BASH_SOURCE:-$(env-home)/$(atb-src)} ; }
atb-vi(){       vi $(atb-source) ; }
atb-env(){      elocal- ; }
atb-usage(){ cat << EOU

AntTweakBar
=============

Glumpy provides a binding to this, hence checking it out.

* http://anttweakbar.sourceforge.net/doc/tools:anttweakbar:download

#. Hmm nasty, lots of windows binaries inside the distribution.


EOU
}
atb-dir(){ echo $(local-base)/env/graphics/atb/AntTweakBar ; }
atb-cd(){  cd $(atb-dir); }
atb-mate(){ mate $(atb-dir) ; }
atb-get(){
   local dir=$(dirname $(atb-dir)) &&  mkdir -p $dir && cd $dir

   local url=http://downloads.sourceforge.net/project/anttweakbar/AntTweakBar_116.zip
   local zip=$(basename $url)

   [ ! -f "$zip" ] && curl -L -O $url
  


}
