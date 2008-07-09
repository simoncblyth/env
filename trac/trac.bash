

trac-usage(){
cat << EOU

   This attempts to automate Trac installation as far as possible
   documentation of the manual installation of 0.11b1 is at 
   
      http://dayabay.phys.ntu.edu.tw/tracs/env/wiki/LeopardTrac

   Considerations...
      0) all packages including package to be installed by a standard procedure
      1) do i need to kickstart the tracitory from a prior backup 


    NODE_TAG      : $NODE_TAG

    trac-version  : $(trac-version)
    trac-instance : $(trac-instance)    the default instance for the node    
    
    TRAC_VERSION  : $TRAC_VERSION
    TRAC_INSTANCE : $TRAC_INSTANCE    
            
                    
    trac-major    : $(trac-major)
    trac-envpath  : $(trac-envpath)
    trac-logpath  : $(trac-logpath)
    trac-inipath  : $(trac-inipath)
    trac-pkgpath  : $(trac-pkgpath)
    
    trac-inheritpath :  $(trac-inheritpath) 
           trac.ini for individual instances in 0.11
           can use an inherit:file:<path> block to use a common conf file
    
    trac-tail    <name>  
    trac-log     <name>
    trac-inicat  <name>
           
           
    For the above the names default to the TRAC_INSTANCE, for the 
    below utilities that take arguments a non default instance must be
    specified thru the environment with eg 
         TRAC_INSTANCE=another  trac-admin-  permission list
         TRAC_INSTANCE=another  trac-configure <a:b:c> ...
    
 
    trac-admin-                    ## NB trailing dash
              trac-admin-      ... into interactive mode
              trac-admin- upgrade        ## db upgrade for new schema   
              trac-admin- permission list 
           
    trac-configure  <block:qty:valu> ... 
           applies edits to  trac.ini by means of triplet arguments
   
 
EOU

}


tracbuild-(){         . $ENV_HOME/trac/tracbuild.bash  && tracbuild-env  $* ; }

bitextra-(){          . $ENV_HOME/trac/package/bitextra.bash  && bitextra-env  $* ; }
tractags-(){          . $ENV_HOME/trac/package/tractags.bash  && tractags-env $* ; }
tracnav-(){           . $ENV_HOME/trac/package/tracnav.bash   && tracnav-env  $* ; }
tractoc-(){           . $ENV_HOME/trac/package/tractoc.bash   && tractoc-env  $* ; }
accountmanager-(){    . $ENV_HOME/trac/package/accountmanager.bash    && accountmanager-env   $* ; }
bitten-(){            . $ENV_HOME/trac/package/bitten.bash    && bitten-env   $* ; }
tractrac-(){          . $ENV_HOME/trac/package/tractrac.bash  && tractrac-env $* ; }
genshi-(){            . $ENV_HOME/trac/package/genshi.bash    && genshi-env   $* ; }
trac2mediawiki-(){    . $ENV_HOME/trac/package/trac2mediawiki.bash    && trac2mediawiki-env   $* ; }

silvercity-(){        . $ENV_HOME/trac/package/silvercity.bash && silvercity-env   $* ; }
pygments-(){          . $ENV_HOME/trac/package/pygments.bash   && pygments-env   $* ; }
docutils-(){          . $ENV_HOME/trac/package/docutils.bash   && docutils-env   $* ; }
textile-(){           . $ENV_HOME/trac/package/textile.bash    && textile-env   $* ; }

bittennotify-(){      . $ENV_HOME/trac/package/bittennotify.bash && bittennotify-env   $* ; }





trac-instance(){
    case ${1:-$NODE_TAG} in
     G) echo workflow ;;
     H) echo env      ;;
     P) echo dybsvn   ;;
     C) echo dybsvn   ;;
     *) echo env      ;;
   esac
}

trac-version(){
   case ${1:-$NODE_TAG} in
     G) echo 0.11rc1 ;;
     H) echo 0.10.4  ;;
     P) echo 0.11    ;;
     C) echo 0.11    ;;
     *) echo 0.11    ;;
   esac
}


trac-env(){
   elocal-
   package-
  
   export TRAC_INSTANCE=$(trac-instance)
   export TRAC_VERSION=$(trac-version)
  
   # these settings ?were? used by svn-apache-* for apache2 config 
   # apache-
   # export TRAC_APACHE2_CONF=$APACHE2_LOCAL/trac.conf 
   #
 
   # when packages need to be installed in a particular order arrange
   # them here ... the rest will be added to the end in alphabetical order
   #
   export TRAC_NAMES_BASE="genshi tractrac bitten" 
   
   trac-ini-
   
}


trac-major(){   echo ${TRAC_VERSION:0:4} ; }
trac-envpath(){ echo $SCM_FOLD/tracs/${1:-$TRAC_INSTANCE} ; }
trac-logpath(){ echo $(trac-envpath $*)/log/trac.log ; }
trac-inipath(){ echo $(trac-envpath $*)/conf/trac.ini ; }
trac-pkgpath(){ echo $ENV_HOME/trac/package ; }

trac-inheritpath(){ echo $SCM_FOLD/conf/trac.ini ; }  


trac-tail(){ tail -f $(trac-logpath $*) ; }
trac-log(){  cd $(dirname $(trac-logpath $*)) ; ls -l  ;}
trac-inicat(){  cat $(trac-inipath $*) ; }


trac-admin-(){   $SUDO trac-admin $(trac-envpath) $* ; }
trac-configure(){ trac-ini-edit $(trac-inipath) $*   ; }



trac-notify-conf(){

  local domain=localhost
  trac-configure notification:smtp_default_domain:$domain notification:smtp_enabled:true

}




