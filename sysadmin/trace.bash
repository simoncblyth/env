# === func-gen- : sysadmin/trace fgp sysadmin/trace.bash fgn trace fgh sysadmin
trace-src(){      echo sysadmin/trace.bash ; }
trace-source(){   echo ${BASH_SOURCE:-$(env-home)/$(trace-src)} ; }
trace-vi(){       vi $(trace-source) ; }
trace-env(){      elocal- ; }
trace-usage(){ cat << EOU





EOU
}
trace-dir(){ echo $(local-base)/env/sysadmin/sysadmin-trace ; }
trace-cd(){  cd $(trace-dir); }
trace-mate(){ mate $(trace-dir) ; }
trace-get(){
   local dir=$(dirname $(trace-dir)) &&  mkdir -p $dir && cd $dir

}

trace-host(){
   local host=$(uname -n)
   echo ${host/.*} 
}
trace-target(){
   case $1 in 
      cms02)  echo belle7.nuu.edu.tw ;; 
      belle7) echo cms02.phys.ntu.edu.tw ;; 
   esac
}
trace-opts(){
   case $1 in 
      cms02) echo -I - ;; 
      belle7) echo -I -T -U ;; 
   esac
}
trace(){
   local msg="== $FUNCNAME :"
   local host=$(trace-host) 
   local opts=$(trace-opts $host)
   local target=$(trace-target $host)
   local opt
   for opt in $opts ; do
      [ "$opt" == "-" ] && opt="" 	    
      local cmd="traceroute $opt $target"
      echo $msg $cmd : $(date)
      eval $cmd
   done	   
}


