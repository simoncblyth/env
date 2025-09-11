# === func-gen- : tools/poco fgp tools/poco.bash fgn poco fgh tools src base/func.bash
poco-source(){   echo ${BASH_SOURCE} ; }
poco-edir(){ echo $(dirname $(poco-source)) ; }
poco-ecd(){  cd $(poco-edir); }
poco-dir(){  echo $LOCAL_BASE/env/tools/poco ; }
poco-cd(){   cd $(poco-dir); }
poco-vi(){   vi $(poco-source) ; }
poco-env(){  elocal- ; }
poco-usage(){ cat << EOU


POCO (Portable Components) C++ Libraries 
==========================================

SCB::

    Looks heavy, corporate and old : like java, boost

    



https://github.com/pocoproject/poco.git

https://pocoproject.org/

https://docs.pocoproject.org/current/00100-GuidedTour.html

https://docs.pocoproject.org/current/00200-GettingStarted.html


EOU
}
poco-get(){
   local dir=$(dirname $(poco-dir)) &&  mkdir -p $dir && cd $dir

}
