

apachebuild-(){ . $ENV_HOME/apache/apachebuild/apachebuild.bash && apachebuild-env $* ; } 
modpython-(){   . $ENV_HOME/apache/apachebuild/modpython.bash && modpython-env $* ; }
mpinfo-(){      . $ENV_HOME/apache/apachebuild/mpinfo.bash      && mpinfo-env $* ; } 

apache-usage(){

   cat << EOU

       which apachectl : $(which apachectl) 

      apache-ls
                ls dso modules
      apache-logs 
                ls logs 
   
      NODE_TAG           : $NODE_TAG
      
      apache-name        : $(apache-name)
      apache-home        : $(apache-home)
      apache-envvars     : $(apache-envvars)
      apache-target      : $(apache-target)
      apache-confdir     : $(apache-confdir)
      apache-modulesdir  : $(apache-modulesdir)
      apache-htdocs      : $(apache-htdocs)
      
      apache-again
           wipes and builds both apache and modpython
           CAUTION:  this wipes installations and rebuilds from the tarball
   
EOU


}

apache-again(){
    apachebuild-again
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
     H) echo $LOCAL_BASE/apache2/$APACHE_NAME ;;
     *) echo $SYSTEM_BASE/apache/$APACHE_NAME ;;
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
   echo $APACHE_HOME/bin/envvars
}

apache-user(){
   case $NODE_TAG in 
     G) echo www ;;
     *) echo apache ;;
   esac
}


apache-confdir(){
  local dir
  case $NODE_APPROACH in
    stock) dir="/private/etc/apache2" ;;
        *) dir="$APACHE_HOME/conf"
  esac
  echo $dir
}

apache-htdocs(){
  echo $APACHE_HOME/htdocs 
}

apache-modulesdir(){
   echo $APACHE_HOME/modules
}

apache-logs(){
  cd $APACHE_HOME/logs
  ls -l 
}


apache-ls(){
   ls -alst $(apache-modulesdir)
}