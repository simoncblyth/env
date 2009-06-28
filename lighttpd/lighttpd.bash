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

lighttpd-base(){   echo /opt/etc/lighttpd ; }
lighttpd-conf(){   echo $(lighttpd-base)/lighttpd.conf ; }
lighttpd-confd(){  echo $(lighttpd-base)/conf.d  ;  }

lighttpd-check(){ 

   /opt/sbin/lighttpd -f $(lighttpd-conf) -p 

}



