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

pkgr-cmds(){ echo yum ipkg port  ; }
pkgr-cmd(){ 
    local pkgr;
    for pkgr in $(pkgr-cmds);
    do
        [ -n "$(which $pkgr 2> /dev/null)" ] && echo $pkgr && return 0;
    done
}


pkgr-prefix-(){
   case $1 in 
     yum) echo -n ;;
    ipkg) echo /opt ;;
    port) echo /opt/local ;;
   esac
}

pkgr-sbin-(){
   case $1 in 
     yum) echo /usr/sbin  ;;
    ipkg) echo /opt/sbin ;;
    port) echo /opt/local/sbin ;;
   esac
}

pkgr-rund-(){
   case $1 in 
     yum) echo /var/run ;;
    port) echo /opt/local/var/run ;;
   esac
}
pkgr-logd-(){
  case $1 in 
   yum) echo /var/log ;;
   port) echo /opt/local/var/log ;;
  esac 
}

pkgr-wwwd-(){
  case $1 in 
    yum) echo /srv/www ;; 
   port) echo /opt/local/www ;;
  esac 
}


pkgr-prefix(){ $FUNCNAME- $(pkgr-cmd) ; }
pkgr-sbin(){   $FUNCNAME- $(pkgr-cmd) ; }
pkgr-logd(){    $FUNCNAME- $(pkgr-cmd) ; }
pkgr-rund(){    $FUNCNAME- $(pkgr-cmd) ; }
pkgr-wwwd(){    $FUNCNAME- $(pkgr-cmd) ; }
