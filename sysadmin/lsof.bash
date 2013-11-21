# === func-gen- : sysadmin/lsof fgp sysadmin/lsof.bash fgn lsof fgh sysadmin
lsof-src(){      echo sysadmin/lsof.bash ; }
lsof-source(){   echo ${BASH_SOURCE:-$(env-home)/$(lsof-src)} ; }
lsof-vi(){       vi $(lsof-source) ; }
lsof-env(){      elocal- ; }
lsof-usage(){ cat << EOU

LSOF
=====

FUNCTIONS
----------

*lsof-processes name*
      list open files for "name" processes 


EOU
}
lsof-dir(){ echo $(local-base)/env/sysadmin/sysadmin-lsof ; }
lsof-cd(){  cd $(lsof-dir); }
lsof-mate(){ mate $(lsof-dir) ; }
lsof-get(){
   local dir=$(dirname $(lsof-dir)) &&  mkdir -p $dir && cd $dir

}


lsof-processes(){
  local name=${1:-mdworker}
  local msg="=== $FUNCNAME :"
  local pid
  local cmd
  pgrep $name | while read pid ; do
     cmd="sudo lsof -p $pid"
     echo $msg $cmd
     eval $cmd
  done 

}
