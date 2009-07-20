# === func-gen- : lighttpd/lighttpd.bash fgp lighttpd/lighttpd.bash fgn lighttpd
lighttpd-src(){      echo lighttpd/lighttpd.bash ; }
lighttpd-source(){   echo ${BASH_SOURCE:-$(env-home)/$(lighttpd-src)} ; }
lighttpd-vi(){       vi $(lighttpd-source) ; }
lighttpd-env(){      pkgr- ; }
lighttpd-usage(){
  cat << EOU
     lighttpd-src : $(lighttpd-src)


    start from the port default :
       g4pb:lighttpd blyth$ sudo cp lighttpd.conf.default lighttpd.conf

EOU
}

lighttpd-check(){   sudo lighttpd -f $(lighttpd-conf) -p  ; }
lighttpd-base(){    echo $(pkgr-prefix)/etc/lighttpd ; }
lighttpd-initd(){   echo $(pkgr-prefix)/etc/init.d ; }
lighttpd-cd(){      cd $(lighttpd-base) ; }

lighttpd-check(){   $(pkgr-sbin)/lighttpd -f $(lighttpd-conf) -p  ; }

lighttpd-conf(){    echo $(lighttpd-base)/lighttpd.conf ; }
lighttpd-confd(){   echo $(lighttpd-base)/conf.d  ;  }
lighttpd-edit(){    sudo vim $(lighttpd-conf) $(lighttpd-confd)/*.conf ; }
lighttpd-ini(){     
  case $(pkgr-cmd) in 
    yum) echo /etc/rc.d/init.d/lighttpd ;;
   ipkg) echo /opt/etc/init.d/S80lighttpd ;; 
   port) echo /opt/local/etc/LaunchDaemons/org.macports.lighttpd/lighttpd.wrapper ;;
  esac 
}

lighttpd-init(){    sudo $(lighttpd-ini) $* ; }
lighttpd-start(){   lighttpd-init start ; }
lighttpd-stop(){    lighttpd-init stop ; }
lighttpd-restart(){ lighttpd-init restart ; }
lighttpd-ps(){     ps aux | grep light ; }


lighttpd-htdocs(){ echo $(pkgr-wwwd)/lighttpd/htdocs ; }

## logging 

lighttpd-logd(){   echo $(pkgr-logd)/lighttpd  ; }
lighttpd-elog(){   echo $(lighttpd-logd)/error.log ; }
lighttpd-alog(){   echo $(lighttpd-logd)/access.log ; }
lighttpd-etail(){  sudo tail -f $(lighttpd-elog) ; } 
lighttpd-atail(){  sudo tail -f $(lighttpd-alog) ; } 


lighttpd-config-(){ cat << EOC

server.document-root        = "$(lighttpd-htdocs)/"

## this matches the pid path in the macports lighttpd.wrapper 
server.pid-file             = "$(pkgr-rund)/lighttpd.pid"
server.errorlog             = "$(lighttpd-elog)"
accesslog.filename          = "$(lighttpd-alog)"

EOC

  [ "$(uname)" == "Darwin" ] && ${FUNCNAME}osx-
}


lighttpd-config-osx-(){ cat << EOX

## set the event-handler (read the performance section in the manual)
server.event-handler = "freebsd-kqueue" # needed on OS X

EOX
}

lighttpd-config(){
   local msg="=== $FUNCNAME :"

   echo $msg manually merge the below with $(lighttpd-conf)
   $FUNCNAME- 

   local dirs="$(lighttpd-logd) $(lighttpd-rund) $(lighttpd-htdocs)"
   echo $msg prepare dirs $dirs 
   local dir
   for dir in $dirs ; do
     [ ! -d "$dir" ] && echo $msg create $dir && sudo mkdir -p $dir 
   done 
}

