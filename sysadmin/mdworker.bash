# === func-gen- : sysadmin/mdworker fgp sysadmin/mdworker.bash fgn mdworker fgh sysadmin
mdworker-src(){      echo sysadmin/mdworker.bash ; }
mdworker-source(){   echo ${BASH_SOURCE:-$(env-home)/$(mdworker-src)} ; }
mdworker-vi(){       vi $(mdworker-source) ; }
mdworker-env(){      elocal- ; lsof- ; }
mdworker-usage(){ cat << EOU

MDWORKER
=========

*mdworker-lsof*
      list open files of processes named mdworker


EOU
}
mdworker-dir(){ echo $(local-base)/env/sysadmin/sysadmin-mdworker ; }
mdworker-cd(){  cd $(mdworker-dir); }
mdworker-mate(){ mate $(mdworker-dir) ; }
mdworker-get(){
   local dir=$(dirname $(mdworker-dir)) &&  mkdir -p $dir && cd $dir

}

mdworker-lsof(){ lsof-processes mdworker ; }
