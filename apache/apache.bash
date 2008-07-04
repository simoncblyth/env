

apachebuild-(){ . $ENV_HOME/apache/apachebuild/apachebuild.bash && apachebuild-env $* ; } 
modpython-(){   . $ENV_HOME/apache/apachebuild/modpython.bash && modpython-env $* ; }
mpinfo-(){      . $ENV_HOME/apache/apachebuild/mpinfo.bash      && mpinfo-env $* ; } 

apache-usage(){

   cat << EOU

       which apachectl : $(which apachectl) 

      apache-ls
                ls libexec
   
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
   ##export APACHE_NAME="httpd-2.0.59"    ## known to work with svn bindings for Trac usage
   ##export APACHE_NAME="httpd-2.0.61"      ## nearest version on the mirror    
   export APACHE_NAME="httpd-2.0.63"      ## nearest version on the mirror    

   export APACHE_HOME=$SYSTEM_BASE/apache/$APACHE_NAME

   [ -d $APACHE_HOME ] && env-prepend $APACHE_HOME/bin

}


apache-target(){
  echo http://cms01.phys.ntu.edu.tw
}

apache-confdir(){
  #echo $APACHE_HOME/etc/apache2
  echo $APACHE_HOME/conf
}

apache-htdocs(){
  echo $APACHE_HOME/htdocs 
}


apache-logs(){
  cd $APACHE_HOME/logs
  ls -l 
}

apache-ls(){
   ls -alst $APACHE_HOME/libexec
}