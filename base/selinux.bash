# === func-gen- : base/selinux fgp base/selinux.bash fgn selinux fgh base
selinux-src(){      echo base/selinux.bash ; }
selinux-source(){   echo ${BASH_SOURCE:-$(env-home)/$(selinux-src)} ; }
selinux-vi(){       vi $(selinux-source) ; }
selinux-env(){      elocal- ; }
selinux-usage(){
  cat << EOU
     selinux-src : $(selinux-src)
     selinux-dir : $(selinux-dir)

    selinux-a2a-- : $(selinux-a2a--)
       print the audit2allow command that will emit the .te commands 

       -m module name
       -l read input since the last reload
       -i input file ... the logfile with the denials

    selinux-a2a
         http://docs.fedoraproject.org//selinux-faq-fc5/#id2961385


EOU
}
selinux-dir(){ echo $(local-base)/env/base/base-selinux ; }
selinux-cd(){  cd $(selinux-dir); }
selinux-mate(){ mate $(selinux-dir) ; }
selinux-get(){
   local dir=$(dirname $(selinux-dir)) &&  mkdir -p $dir && cd $dir

}


selinux-log(){
   case $NODE_TAG in
     N) echo /var/log/audit/audit.log ;;
     *) echo /var/log/messages ;;
   esac
}


selinux-httpd_can_network_connect(){
   ## avoid "name_connect" AVC 
    sudo setsebool httpd_can_network_connect 1 
}


selinux-a2w(){
  type $FUNCNAME
  sudo echo just getting sudofied  ... as accepting passwords messes up the pipe on the next line 
  sudo cat $(selinux-log) | sudo audit2why
}


selinux-a2a--(){  cat << EOC
sudo audit2allow -m local -l -i $(selinux-log)
EOC
}
selinux-a2a-(){
   #$FUNCNAME-
   eval $($FUNCNAME-)
}

selinux-a2a(){
   local tmp=/tmp/env/$FUNCNAME && mkdir -p $tmp 
   $FUNCNAME- > $tmp/local.te
}



