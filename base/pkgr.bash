# === func-gen- : base/pkgr.bash fgp base/pkgr.bash fgn pkgr
pkgr-src(){      echo base/pkgr.bash ; }
pkgr-source(){   echo ${BASH_SOURCE:-$(env-home)/$(pkgr-src)} ; }
pkgr-vi(){       vi $(pkgr-source) ; }
pkgr-env(){      elocal- ; }
pkgr-usage(){
  cat << EOU
     pkgr-src : $(pkgr-src)

     pkgr-cmds : $(pkgr-cmds)
     pkgr-cmd : $(pkgr-cmd)
     pkgr-prefix : $(pkgr-prefix)



EOU
}

pkgr-cmds(){ echo yum ipkg ; }
pkgr-cmd(){ 
    local pkgr;
    for pkgr in $(pkgr-cmds);
    do
        [ -n "$(which $pkgr 2> /dev/null)" ] && echo $pkgr && return 0;
    done
}


pkgr-prefix(){
   local cmd=$(pkgr-cmd)
   case $cmd in 
     yum) echo -n ;;
    ipkg) echo /opt ;;
   esac
}



