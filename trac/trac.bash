

trac-usage(){
cat << EOU

   This attempts to automate Trac installation as far as possible
   documentation of the manual installation of 0.11b1 is at 
   
      http://dayabay.phys.ntu.edu.tw/tracs/env/wiki/LeopardTrac

   Considerations...
      0) all packages including package to be installed by a standard procedure
      1) do i need to kickstart the tracitory from a prior backup 


    trac-tail <name>  
    trac-logpath <name>

        
    
    
    trac-inipath <name>    
    trac-inicat  <name>
           utilities targeted to the named instance, defaulting to TRAC_INSTANCE


   



    trac-inheritpath 
           trac.ini for individual instances can use an inherit:file:<path> block to use
           a common conf file
    
   
               
               
   Utilities targeted to the default instance for the NODE_TAG , to use another :
       TRAC_INSTANCE=another  trac-blah blah      
         
    trac-admin-                 ## NB trailing dash
              trac-admin-      ... into interactive mode
              trac-admin- upgrade        ## db upgrade for new schema   
              trac-admin- permission list 
           
    trac-configure  <block:qty:valu> ... 
           applies edits to  trac.ini by means of triplet arguments
           


    Distributed commands over the packages... 

       trac-names  : names of all packages 
       trac-auto   : \$name-auto 
       trac-status : \$name-status 
    
    
 
EOU

}

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



trac-inheritpath(){   echo  $SCM_FOLD/conf/trac.ini ; }  ## is inherit 0.11 only ?  

trac-env(){
   elocal-
   package-
  
   
   case $NODE_TAG in
     G) export TRAC_INSTANCE=workflow ; export TRAC_VERSION=0.11rc1 ;;
     H) export TRAC_INSTANCE=env      ; export TRAC_VERSION=0.10.4  ;;
     *) export TRAC_INSTANCE=    ;;
   esac

   ## these settings are used by svn-apache-* for apache2 config 
   apache2-
   export TRAC_APACHE2_CONF=$APACHE2_LOCAL/trac.conf 
   export TRAC_EGG_CACHE=/tmp/trac-egg-cache

}




trac-major(){  echo ${TRAC_VERSION:0:4} ; }


trac-tail(){ tail -f $(trac-logpath $*) ; }
trac-log(){  cd $(dirname $(trac-logpath $*)) ; ls -l  ;}

trac-logpath(){
  local name=${1:-$TRAC_INSTANCE}
  echo $SCM_FOLD/tracs/$name/log/trac.log
}

trac-inipath(){
  local name=${1:-$TRAC_INSTANCE}
  echo $SCM_FOLD/tracs/$name/conf/trac.ini
}

trac-inicat(){
  cat $(trac-inipath $*) 
}





trac-admin-(){
   local name=$TRAC_INSTANCE
   $SUDO trac-admin $SCM_FOLD/tracs/$name $*
}

trac-configure(){
   local msg="=== $FUNCNAME :"
   local name=$TRAC_INSTANCE
   #shift
   local tini=$SCM_FOLD/tracs/$name/conf/trac.ini 
   trac-ini-
   
   echo $msg editing $tini with $*
   trac-ini-edit $tini $*
   
}



trac-names(){
   local iwd=$PWD
   cd $ENV_HOME/trac/package   
   for bash in *.bash
   do
      local name=${bash/.bash/}
      echo $name
   done
   cd $iwd
}

trac-diff(){      trac-f diff ; }
trac-status(){    trac-f status ; }
trac-summary(){   trac-f summary ;  }
trac-makepatch(){ trac-f makepatch ;  }

trac-f(){
  local msg="=== $FUNCNAME :"
  local f=$1
  for name in $(trac-names)
  do
      $name-   ||  (  echo $msg ABORT you must define the precursor $name- in trac/trac.bash && sleep 100000 )
      package-$f $name
  done
}






trac-notify-conf(){

  local domain=localhost
  trac-configure notification:smtp_default_domain:$domain notification:smtp_enabled:true

}




