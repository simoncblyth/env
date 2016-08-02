# === func-gen- : juno/sniper/sniper fgp juno/sniper/sniper.bash fgn sniper fgh juno/sniper
sniper-src(){      echo juno/sniper/sniper.bash ; }
sniper-source(){   echo ${BASH_SOURCE:-$(env-home)/$(sniper-src)} ; }
sniper-vi(){       vi $(sniper-source) ; }
sniper-env(){      elocal- ; }
sniper-usage(){ cat << EOU





EOU
}
sniper-dir(){ echo $HOME/sniper ; }
sniper-cd(){  cd $(sniper-dir); }
sniper-get(){
   local dir=$(dirname $(sniper-dir)) &&  mkdir -p $dir && cd $dir

   [ ! -d sniper ] && svn co http://juno.ihep.ac.cn/svn/sniper/trunk sniper
}
