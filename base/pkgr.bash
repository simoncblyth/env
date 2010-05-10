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

pkgr-bin-(){
   case $1 in 
     yum) echo /usr/bin  ;;
    ipkg) echo /opt/bin ;;
    port) echo /opt/local/bin ;;
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
   ipkg) echo /opt/var/log ;;
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
pkgr-bin(){    $FUNCNAME- $(pkgr-cmd) ; }
pkgr-sbin(){   $FUNCNAME- $(pkgr-cmd) ; }
pkgr-logd(){    $FUNCNAME- $(pkgr-cmd) ; }
pkgr-rund(){    $FUNCNAME- $(pkgr-cmd) ; }
pkgr-wwwd(){    $FUNCNAME- $(pkgr-cmd) ; }


pkgr-ln-(){
  local msg="=== $FUNCNAME :"
  local opp=${1:-ln}
  local src=${2:-python2.5}
  local tgt=${3:-python}

  local bin=$(pkgr-bin)
  local iwd=$PWD
  cd $bin
  local cmd

  if [ "$opp" == "ln" ]; then 

      if [ ! -L "$tgt" ]; then
         echo $msg create new link target \"$tgt\" exposing src \"$src\"  
         cmd="sudo ln -s $src $tgt"
      else
         local now=$(readlink $tgt) 
         [ "$now" == "$src" ] && echo $msg target $tgt is already exposing src $src ... nothing to do && return 0 
         cmd="sudo ln -s $src $tgt"
      fi

  elif [ "$opp" == "uln" ]; then

      [ ! -L "$tgt" ] && echo $msg no such target $tgt ... nothing to do  && return 0
      local now=$(readlink $tgt) 
      [ "$now" == "$src" ] && echo $msg removing target \"$tgt\" that is exposing src \"$src\" ...  
      [ ${#tgt} -lt 5 ] && echo $msg sanity length check on tgt \"$tgt\" fails ... aborting && return 1
      cmd="sudo rm $tgt"

      read -p "$msg enter YES to proceed with removal of the target ...  \"$cmd\" " ans
      [ "$ans" != "YES" ] && echo $msg skipping && return 0
      
  fi 

  echo $msg $cmd 
  eval $cmd 

  cd $iwd
}

pkgr-ln(){ pkgr-ln- ln $* ; }
pkgr-uln(){ pkgr-ln- uln $* ; }
