

 ## DEPRECATED THIS IS BEING MIGRATED TO trac/trac.bash 


 ## new style... reduce env pollution and startup time 

 silvercity-(){        . $ENV_HOME/scm/trac/silvercity.bash ; }
 docutils-(){          . $ENV_HOME/scm/trac/docutils.bash   ; }

 trac2mediawiki-(){    . $ENV_HOME/scm/trac/trac2mediawiki.bash   ; }
 trac2latex-(){        . $ENV_HOME/scm/trac/trac2latex.bash   ; }
 traclxml-(){          . $ENV_HOME/scm/trac/traclxml.bash   ; } 
 tractoc-(){           . $ENV_HOME/scm/trac/tractoc.bash   ; } 
 tracxsltmacro-(){     . $ENV_HOME/scm/trac/tracxsltmacro.bash   ; }
 traclegendbox-(){     . $ENV_HOME/scm/trac/traclegendbox.bash   ; }
 tracincludemacro-(){  . $ENV_HOME/scm/trac/tracincludemacro.bash   ; }
 db2trac-(){           . $ENV_HOME/scm/trac/db2trac.bash   ; }
 tracenv-(){           . $ENV_HOME/scm/trac/tracenv.bash   ; }
 hepreztrac-(){        . $ENV_HOME/scm/trac/hepreztrac.bash   ; }

 tracxmlrpc-(){        . $ENV_HOME/scm/trac/tracxmlrpc.bash ; }
 trachttpauth-(){      . $ENV_HOME/scm/trac/trachttpauth.bash ; }


otrac-env(){

   elocal-
   apache2-
   python-
   
   otrac-base

}

otrac-base(){

  export TRAC_NAME=trac-0.10.4
  TRAC_NIK=trac

  export TRAC_HOME=$LOCAL_BASE/$TRAC_NIK
  export TRAC_COMMON=$TRAC_HOME/common

 # export TRAC_APACHE2_CONF=$APACHE2_LOCAL/trac.conf 
 # export TRAC_EGG_CACHE=/tmp/trac-egg-cache

  export TRAC_ENV_XMLRPC="http://$USER:$(private-val NON_SECURE_PASS)@$SCM_HOST:$SCM_PORT/tracs/$SCM_TRAC/login/xmlrpc"

  export TRAC_SHARE_FOLD=$PYTHON_HOME/share/trac




}



otrac-kitchensink(){

 local trac_iwd=$(pwd)
 
 cd $ENV_HOME/scm/trac

 [ -r trac-conf.bash ]                   && . trac-conf.bash
      
 ## caution webadmin is a prerequisite to accountmanager      
      
 [ -r trac-plugin-webadmin.bash ]        && . trac-plugin-webadmin.bash       
 [ -r trac-plugin-accountmanager.bash ]  && . trac-plugin-accountmanager.bash 

#  move to new style...  [ -r trac-plugin-tracnav.bash ]         && . trac-plugin-tracnav.bash 

 [ -r trac-plugin-restrictedarea.bash ]  && . trac-plugin-restrictedarea.bash
 [ -r trac-plugin-pygments.bash ]        && . trac-plugin-pygments.bash     
    
 [ -r trac-macro-latexformulamacro.bash ] && . trac-macro-latexformulamacro.bash  
 [ -r trac-plugin-reposearch.bash ]       && . trac-plugin-reposearch.bash           
                    
                                    
 [ -r trac-build.bash ]                   && . trac-build.bash          ## depends on clearsilver  

## trac-build-(){  [ -r $TRAC_HOME/trac-build.bash ]  && . $TRAC_HOME/trac-build.bash ; }



## confusing approach 
## db2trac-(){  db2trac ;  [ -r $(db2trac-dir)/db2trac.bash ] && . $(db2trac-dir)/db2trac.bash ; }



#[ -r trac-test.bash ]      && . trac-test.bash

 ## caution must exit with initial directory
 cd $trac_iwd
 
} 
 
 



otrac-ini(){
  local name=${1:-$SCM_TRAC}
  $SUDO vi  $SCM_FOLD/tracs/$name/conf/trac.ini
}



