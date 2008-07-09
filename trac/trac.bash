

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
           
      
    trac-instances  : $(trac-instances)
          names of all the instances from looking in $SCM_FOLD/tracs
        
                
                          
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
     P) echo env      ;;
     C) echo env      ;;
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
   export TRAC_USER=$(trac-user)
  
   # these settings ?were? used by svn-apache-* for apache2 config 
   # apache-
   # export TRAC_APACHE2_CONF=$APACHE2_LOCAL/trac.conf 
   #
 
   # when packages need to be installed in a particular order arrange
   # them here ... the rest will be added to the end in alphabetical order
   #
   export TRAC_NAMES_BASE="genshi tractrac bitten" 
   
   
   
}


trac-user(){
   apache-
   echo $(apache-user)
}


trac-major(){   echo ${TRAC_VERSION:0:4} ; }
trac-envpath(){ echo $SCM_FOLD/tracs/${1:-$TRAC_INSTANCE} ; }
trac-repopath(){ echo $SCM_FOLD/repos/${1:-$TRAC_INSTANCE} ; }
trac-logpath(){ echo $(trac-envpath $*)/log/trac.log ; }
trac-inipath(){ echo $(trac-envpath $*)/conf/trac.ini ; }
trac-pkgpath(){ echo $ENV_HOME/trac/package ; }

trac-inheritpath(){ echo $SCM_FOLD/conf/trac.ini ; }  


trac-tail(){ tail -f $(trac-logpath $*) ; }
trac-log(){  cd $(dirname $(trac-logpath $*)) ; ls -l  ;}
trac-inicat(){  cat $(trac-inipath $*) ; }
trac-inhcat(){  cat $(trac-inheritpath $*) ; }

trac-admin-(){   $SUDO trac-admin $(trac-envpath) $* ; }
trac-configure(){ trac-edit-ini $(trac-inipath) $*   ; }
trac-edit-ini(){

   local path=$1
   local user=$TRAC_USER
   shift
 
   sudo perl $ENV_HOME/base/ini-edit.pl $path $*  
   sudo chown $user:$user $path
}


trac-notify-conf(){

  local domain=localhost
  trac-configure notification:smtp_default_domain:$domain notification:smtp_enabled:true

}


trac-instances(){
   local iwd=$PWD
   cd $SCM_FOLD/tracs
   for name in $(ls -1)
   do
      [ -d $name ] && echo $name
   done
   cd $iwd
}



trac-prepare(){

   trac-inherit-setup
   trac-upgrade

}



trac-inherit-setup(){

   [ "$(trac-major)" != "0.11" ] && echo $msg this is only relevant to 0.11 && return 1
   
    local msg="=== $FUNCNAME :"
    local inherit=$(trac-inheritpath)
    local dir=$(dirname $inherit)
    local user=$TRAC_USER
    
    [ ! -d $dir ] && echo $msg creating dir $dir for global inherited conf && sudo mkdir -p $dir && sudo chown $user:$user $dir
    
    echo $msg planting the inherit reference in all the instances 
    for name in $(trac-instances)
    do
       TRAC_INSTANCE=$name trac-configure inherit:file:$inherit
    done
    
    [ ! -f $inherit ] && echo $msg bootstraping global config $inherit && trac-inherit 
    

}

trac-inherit(){

   local live=$(trac-inheritpath)
   local path=${1:-$live}
   local tmp=/tmp/env/${FUNCNAME/-*/} && mkdir -p $tmp
   local name=$(basename $path)
   local user=$TRAC_USER
   
   trac-inherit- > $tmp/$name
   
   if [ "$path" == "$live" ]; then
      sudo cp $tmp/$name $live 
      sudo chown $user:$user $live
   fi

}


trac-inherit-(){
 
   cat << EOI

##  http://trac-hacks.org/wiki/TagsPlugin

[components]
tractags.* = enabled
acct_mgr.admin.accountmanageradminpage = enabled
acct_mgr.api.accountmanager = enabled
acct_mgr.htfile.htdigeststore = disabled
acct_mgr.htfile.htpasswdstore = enabled
acct_mgr.web_ui.accountmodule = enabled
acct_mgr.web_ui.loginmodule = enabled
acct_mgr.web_ui.registrationmodule = disabled
includemacro.* = enabled
latexdummy.* = enabled
latexmacro.* = enabled
mwimagemacro.* = enabled
mwmediawikicmdmacro.* = enabled
mwmediawikienddocmacro.* = enabled
mwmediawikigetresmacro.* = enabled
mwmediawikimacro.* = enabled
mwmediawikispecialcharmacro.* = enabled
mwmediawikitablemacro.* = enabled
mwmediawikitimestampmacro.* = enabled
mwtracnavmacro.* = enabled
other.* = enabled
teximagemacro.* = enabled
texlatexbasicheadersmacro.* = enabled
texlatexbegindocmacro.* = enabled
texlatexcmdmacro.* = enabled
texlatexdocclsmacro.* = enabled
texlatexenddocmacro.* = enabled
texlatexgetresmacro.* = enabled
texlatexmacro.* = enabled
texlatexspecialcharmacro.* = enabled
texlatextablemacro.* = enabled
texlatexusepkgmacro.* = enabled
textimestampmacro.* = enabled
trac.web.auth.loginmodule = disabled
##
##remove for tractags 0.6 
##trac.wiki.web_ui.wikimodule = disabled
##
trac2latex.* = enabled
trac2mediawiki.* = enabled
tracnav.* = enabled
tracreposearch.* = enabled
tractoc.* = enabled
webadmin.* = enabled
xslt.* = enabled



   
   
EOI

}



trac-upgrade(){

    local msg="=== $FUNCNAME :"
    local user=$TRAC_USER
    
    svn-
    local authz=$(svn-authzpath)
    for name in $(trac-instances)
    do
       local path=$(trac-inipath $name)
       echo $msg commenting default_handler TagsWikiModule setting from $path user $user
       sudo perl -pi -e "s,^(default_handler = TagsWikiModule),#\$1 ## removed by $BASH_SOURCE::$FUNCNAME ,  " $path
       sudo chown $user:$user $path
       
       TRAC_INSTANCE=$name trac-configure trac:repository_dir:$(trac-repopath) trac:authz_file:$authz
       
    done
}




