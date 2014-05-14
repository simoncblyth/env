# === func-gen- : nuwa/nuwacmt fgp nuwa/nuwacmt.bash fgn nuwacmt fgh nuwa
nuwacmt-src(){      echo nuwa/nuwacmt.bash ; }
nuwacmt-source(){   echo ${BASH_SOURCE:-$(env-home)/$(nuwacmt-src)} ; }
nuwacmt-vi(){       vi $(nuwacmt-source) ; }
nuwacmt-env(){      elocal- ; }
nuwacmt-usage(){ cat << EOU





EOU
}
nuwacmt-dir(){ echo $(local-base)/env/nuwa/nuwa-nuwacmt ; }
nuwacmt-cd(){  cd $(nuwacmt-dir); }
nuwacmt-mate(){ mate $(nuwacmt-dir) ; }
nuwacmt-get(){
   local dir=$(dirname $(nuwacmt-dir)) &&  mkdir -p $dir && cd $dir

}


nuwacmt-config(){
   local msg="=== $FUNCNAME :"
   local pkg=$1
   shift
   [ ! -d "$pkg/cmt" ] && echo ERROR NO cmt SUBDIR && sleep 1000000
   local iwd=$PWD

   echo $msg for pkg $pkg
   cd $pkg/cmt

   cmt config

   . setup.sh 

   cd $iwd
}

nuwacmt-showpath(){
   local var=${1:-LD_LIBRARY_PATH}
   local val
   eval val=\$$var
   echo $msg $var
   echo $val | tr ":" "\n"
}

nuwacmt-info(){
   nuwacmt-showpath PATH
   nuwacmt-showpath DYLD_LIBRARY_PATH
   nuwacmt-showpath LD_LIBRARY_PATH
}



nuwacmt-lslib(){
   ls -l $DYB/NuWa-trunk/dybgaudi/InstallArea/$CMTCONFIG/lib
}


