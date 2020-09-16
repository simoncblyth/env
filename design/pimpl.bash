# === func-gen- : design/pimpl fgp design/pimpl.bash fgn pimpl fgh design src base/func.bash
pimpl-source(){   echo ${BASH_SOURCE} ; }
pimpl-edir(){ echo $(dirname $(pimpl-source)) ; }
pimpl-ecd(){  cd $(pimpl-edir); }
pimpl-dir(){  echo $LOCAL_BASE/env/design/pimpl ; }
pimpl-cd(){   cd $(pimpl-dir); }
pimpl-vi(){   vi $(pimpl-source) ; }
pimpl-env(){  elocal- ; }
pimpl-usage(){ cat << EOU

Pimpl and alternatives for dependency control/flexibility
===========================================================



* https://en.wikipedia.org/wiki/Opaque_pointer

  C : implementing struct defined in .c not .h 



Opaque Pointer

* https://www.geeksforgeeks.org/opaque-pointer/

  windows/apple example



* The Pimpl Pattern

  https://www.bfilipek.com/2018/01/pimpl.html

* In-depth: PIMPL vs pure virtual interfaces

  https://www.gamasutra.com/view/news/167098/Indepth_PIMPL_vs_pure_virtual_interfaces.php


EOU
}
pimpl-get(){
   local dir=$(dirname $(pimpl-dir)) &&  mkdir -p $dir && cd $dir

}
