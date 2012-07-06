otrac-vi(){ vi $BASH_SOURCE ; }
otrac-usage(){ cat << EOU

EOU
}
 ## DEPRECATED THIS IS BEING MIGRATED TO trac/trac.bash 


 ## new style... reduce env pollution and startup time 

 osilvercity-(){        . $ENV_HOME/otrac/silvercity.bash ; }
 odocutils-(){          . $ENV_HOME/otrac/docutils.bash   ; }

 otrac2mediawiki-(){    . $ENV_HOME/otrac/trac2mediawiki.bash   ; }
 otrac2latex-(){        . $ENV_HOME/otrac/trac2latex.bash   ; }
 otraclxml-(){          . $ENV_HOME/otrac/traclxml.bash   ; } 
 otractoc-(){           . $ENV_HOME/otrac/tractoc.bash   ; } 
 otracxsltmacro-(){     . $ENV_HOME/otrac/tracxsltmacro.bash   ; }
 otraclegendbox-(){     . $ENV_HOME/otrac/traclegendbox.bash   ; }
 otracincludemacro-(){  . $ENV_HOME/otrac/tracincludemacro.bash   ; }
 odb2trac-(){           . $ENV_HOME/otrac/db2trac.bash   ; }
 otracenv-(){           . $ENV_HOME/otrac/tracenv.bash   ; }
 ohepreztrac-(){        . $ENV_HOME/otrac/hepreztrac.bash   ; }

 otrachttpauth-(){      . $ENV_HOME/otrac/trachttpauth.bash ; }

# tracxmlrpc- moved to env.bash
  

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



