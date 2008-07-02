

apachebuild-(){ . $ENV_HOME/apache/apachebuild/apachebuild.bash && apachebuild-env $* ; } 

apache-env(){
   ##export APACHE_NAME="httpd-2.0.59"    ## known to work with svn bindings for Trac usage
   export APACHE_NAME="httpd-2.0.61"      ## nearest version on the mirror    

}

