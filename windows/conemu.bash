# === func-gen- : windows/conemu fgp windows/conemu.bash fgn conemu fgh windows
conemu-src(){      echo windows/conemu.bash ; }
conemu-source(){   echo ${BASH_SOURCE:-$(env-home)/$(conemu-src)} ; }
conemu-vi(){       vi $(conemu-source) ; }
conemu-env(){      elocal- ; }
conemu-usage(){ cat << EOU



* http://stackoverflow.com/questions/20202269/set-up-git-bash-to-work-with-tabs-on-windows

* http://conemu.github.io/

* http://superuser.com/questions/454380/git-bash-here-in-conemu




EOU
}
conemu-dir(){ echo $(local-base)/env/windows/windows-conemu ; }
conemu-cd(){  cd $(conemu-dir); }
conemu-mate(){ mate $(conemu-dir) ; }
conemu-get(){
   local dir=$(dirname $(conemu-dir)) &&  mkdir -p $dir && cd $dir

}
