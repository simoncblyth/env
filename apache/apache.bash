

apachebuild-(){ . $ENV_HOME/apache/apachebuild/apachebuild.bash && apachebuild-env $* ; } 
apacheconf-(){  . $ENV_HOME/apache/apacheconf/apacheconf.bash   && apacheconf-env $* ; } 
modpython-(){   . $ENV_HOME/apache/apachebuild/modpython.bash   && modpython-env $* ; }
mpinfo-(){      . $ENV_HOME/apache/apachebuild/mpinfo.bash      && mpinfo-env $* ; } 
iptables-(){    . $ENV_HOME/apache/apacheconf/iptables.bash     && iptables-env $* ; }



apache-usage(){

   cat << EOU

       which apachectl : $(which apachectl) 

      apache-ls
                ls dso modules
      apache-logs 
                ls logs 
   
      NODE_TAG           : $NODE_TAG
      
      NB where it is likely to be useful to grab the 
      value for another node ... its best to implement 
      in a fully functional manner.. allowing simple access
      like 
          remval=$(NODE_TAG=R apache-confdir)
      
      rather than fooling with apache- to set the exports 
      
      
      apache-name        : $(apache-name)
      apache-user        : $(apache-user)
      apache-home        : $(apache-home)
      apache-envvars     : $(apache-envvars)
      apache-target      : $(apache-target)
      apache-confdir     : $(apache-confdir)
      apache-modulesdir  : $(apache-modulesdir)
      apache-htdocs      : $(apache-htdocs)
      apache-logdir      : $(apache-logdir)
      
      apache-sudouser    : $(apache-sudouser)
      
      apache-again
           wipes and builds both apache and modpython
           CAUTION:  this wipes installations and rebuilds from the tarball
   
   
      Precursors ...
      
         apachebuild-
         apacheconf-
         modpython-
         mpinfo-
         iptables-
         
     
   
   
EOU


}

apache-again(){
    
    apachebuild-
    apachebuild-again
    
    modpython-
    modpython-again

}


apache-env(){

   elocal-
   export APACHE_NAME=$(apache-name)
   export APACHE_HOME=$(apache-home)
   
   [ -d $APACHE_HOME ] && env-prepend $APACHE_HOME/bin

}

apache-home(){
   case $NODE_TAG in 
     H) echo $(local-base)/apache2/$(apache-name) ;;
     *) echo $(local-system-base)/apache/$(apache-name) ;;
   esac
}

apache-name(){

 #
 # httpd-2.0.59  known to work with svn bindings for Trac usage
 # httpd-2.0.61  nearest version on the mirror    
 # httpd-2.0.63  nearest version on the mirror    
 #

   case $NODE_TAG in 
      H) echo httpd-2.0.59 ;;
      *) echo httpd-2.0.63 ;;
   esac
}

apache-target(){
  echo http://cms01.phys.ntu.edu.tw
}

apache-envvars(){
  case $NODE_TAG in 
   H) echo $APACHE_HOME/sbin/envvars ;;
   *) echo $APACHE_HOME/bin/envvars ;;
  esac 
}

apache-user(){
   case $NODE_TAG in 
     G) echo www ;;
     C) echo nobody ;;
     *) echo apache ;;
   esac
}

apache-confdir(){
  case $NODE_TAG in
        G) echo /private/etc/apache2 ;;
        H) echo $(apache-home)/etc/apache2 ;;
        C) echo $(apache-home)/conf ;;
        *) echo $(apache-home)/conf ;;
  esac
}

apache-htdocs(){
  
  case $NODE_TAG in 
    G) echo /Library/WebServer/Documents ;;
    H) echo $(apache-home)/share/apache2/htdocs ;;
    *) echo $(apache-home)/htdocs  ;;
  esac  
}

apache-modulesdir(){
  case $NODE_TAG in 
     G) echo /usr/libexec/apache2 ;;
     H) echo $APACHE_HOME/libexec ;;
     *) echo $APACHE_HOME/modules ;;
  esac    
}

apache-logdir(){
   case $NODE_TAG in 
      G) echo /var/log/apache2 ;;
      H) echo $APACHE_HOME/var/apache2/log ;;
      *) echo $APACHE_HOME/logs ;;
   esac   
}   
 
apache-conf(){
   echo $(apache-confdir)/httpd.conf
}   
     
apache-vi(){
   vi $(apache-conf)
}
         
  
apache-sudouser(){
  local user=$(apache-user)
  [ -n "$SUDO" ] && echo $SUDO -u $user || echo ""
}
     
apache-addline(){

  local msg="=== $FUNCNAME :"
  local line=$1
  local conf=$(apache-conf)
  local sudouser=$(apache-sudouser)

  grep -q "$line" $conf && echo $msg line \"$line\" already present in $conf  || $sudouser echo "$line" >> $conf  

}
   
   
   
apache-logs(){
  cd $(apache-logdir)
  ls -l 
}

apache-ls(){
   ls -alst $(apache-modulesdir)
}