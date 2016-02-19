# === func-gen- : vr/vr fgp vr/vr.bash fgn vr fgh vr
vr-src(){      echo vr/vr.bash ; }
vr-source(){   echo ${BASH_SOURCE:-$(env-home)/$(vr-src)} ; }
vr-vi(){       vi $(vr-source) ; }
vr-env(){      elocal- ; }
vr-usage(){ cat << EOU

VR Interface Dev Experience
============================


* "Wild West of VR - Discovering the Rules of Oculus Rift Development"

* https://www.youtube.com/watch?v=_vqNpZqnl1o&index=27&list=PLckFgM6dUP2hc4iy-IdKFtqR9TeZWMPjm









EOU
}
vr-dir(){ echo $(local-base)/env/vr/vr-vr ; }
vr-cd(){  cd $(vr-dir); }
vr-mate(){ mate $(vr-dir) ; }
vr-get(){
   local dir=$(dirname $(vr-dir)) &&  mkdir -p $dir && cd $dir

}
