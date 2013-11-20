# === func-gen- : sysadmin/dns fgp sysadmin/dns.bash fgn dns fgh sysadmin
dns-src(){      echo sysadmin/dns.bash ; }
dns-source(){   echo ${BASH_SOURCE:-$(env-home)/$(dns-src)} ; }
dns-vi(){       vi $(dns-source) ; }
dns-env(){      elocal- ; }
dns-usage(){ cat << EOU

DNS
====

* http://www.rackspace.com/knowledge_center/article/changing-dns-settings-on-linux

FUNCTIONS
----------

*dns-edit*
        takes immediate effect, no need to reboot



EOU
}
dns-dir(){ echo $(local-base)/env/sysadmin/sysadmin-dns ; }
dns-cd(){  cd $(dns-dir); }
dns-mate(){ mate $(dns-dir) ; }
dns-get(){
   local dir=$(dirname $(dns-dir)) &&  mkdir -p $dir && cd $dir
}

dns-edit(){
   local msg="=== $FUNCNAME :"
   local cmd="sudo vi /etc/resolv.conf"
   echo $msg $cmd
   eval $cmd
}

dns-test(){
   time curl -s http://www.google.com
}


