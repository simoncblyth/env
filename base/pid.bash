# === func-gen- : base/pid fgp base/pid.bash fgn pid
pid-src(){      echo base/pid.bash ; }
pid-source(){   echo ${BASH_SOURCE:-$(env-home)/$(pid-src)} ; }
pid-vi(){       vi $(pid-source) ; }
pid-env(){      elocal- ; }
pid-usage(){
  cat << EOU
     pid-src : $(pid-src)

EOU
}


pid-path(){ echo /proc/$1/status ; }

pid-vmsize-(){
   local pid=$1
   local path=$(pid-path $pid)
   while true
   do
       [ -f $path ] && grep VmSize $path || return 0
       sleep 5
   done
}



