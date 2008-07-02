

apachebuild-(){ . $ENV_HOME/apache/apachebuild/apachebuild.bash && apachebuild-env $* ; } 

apache-env(){

   elocal-
   ##export APACHE_NAME="httpd-2.0.59"    ## known to work with svn bindings for Trac usage
   ##export APACHE_NAME="httpd-2.0.61"      ## nearest version on the mirror    
   export APACHE_NAME="httpd-2.0.63"      ## nearest version on the mirror    

   export APACHE_HOME=$SYSTEM_BASE/apache/$APACHE_NAME

}

