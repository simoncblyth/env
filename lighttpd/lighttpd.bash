# === func-gen- : lighttpd/lighttpd.bash fgp lighttpd/lighttpd.bash fgn lighttpd
lighttpd-src(){      echo lighttpd/lighttpd.bash ; }
lighttpd-source(){   echo ${BASH_SOURCE:-$(env-home)/$(lighttpd-src)} ; }
lighttpd-vi(){       vi $(lighttpd-source) ; }
lighttpd-env(){      elocal- ; }
lighttpd-usage(){
  cat << EOU
     lighttpd-src : $(lighttpd-src)

EOU
}

lighttpd-cd(){      cd $(lighttpd-base) ; }

lighttpd-check(){   /opt/sbin/lighttpd -f $(lighttpd-conf) -p  ; }
lighttpd-base(){    echo /opt/etc/lighttpd ; }
lighttpd-initd(){   echo /opt/etc/init.d ; }

lighttpd-conf(){    echo $(lighttpd-base)/lighttpd.conf ; }
lighttpd-confd(){   echo $(lighttpd-base)/conf.d  ;  }
lighttpd-edit(){    sudo vim $(lighttpd-conf) $(lighttpd-confd)/*.conf ; }
lighttpd-ini(){     echo $(lighttpd-initd)/S80lighttpd ; }
lighttpd-init(){    sudo $(lighttpd-ini) $* ; }

lighttpd-start(){   lighttpd-init start ; }
lighttpd-stop(){    lighttpd-init stop ; }
lighttpd-restart(){ lighttpd-init restart ; }

lighttpd-ps(){     ps aux | grep light ; }

