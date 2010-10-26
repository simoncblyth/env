# === func-gen- : base/du fgp base/du.bash fgn du fgh base
du-src(){      echo base/du.bash ; }
du-source(){   echo ${BASH_SOURCE:-$(env-home)/$(du-src)} ; }
du-vi(){       vi $(du-source) ; }
du-env(){      elocal- ; }
du-usage(){
  cat << EOU
     du-src : $(du-src)
     du-dir : $(du-dir)


EOU
}
du-dir(){ echo $(local-base)/env/base/base-du ; }
du-cd(){  cd $(du-dir); }
du-mate(){ mate $(du-dir) ; }
du-get(){
   local dir=$(dirname $(du-dir)) &&  mkdir -p $dir && cd $dir

}


du-dirs(){
   local d
   ls -1 ${1:-/} | while read d ; do
      case $d in
         home|data|boot) echo skip ... $d ;;
                      *) sudo du -hs $d  ;; 
      esac
   done 
}


