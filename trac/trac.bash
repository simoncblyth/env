

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
    trac-vi      <name>   
           for name of ".." this edits the common inherited config  
                
      
    trac-instances  : "$(trac-instances)"
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
   
 

    trac-logging 
          this relies on local patch mods, see 
            http://dayabay.phys.ntu.edu.tw/tracs/env/wiki/TracTweaking#traclogs    
 
 
 
 
    $(type trac-prepare)
    
    
    
    trac-admin-upgrade
         do trac-admin "upgrade" and "wiki upgrade" to 
         bring 0.10.4 instances up to 0.11 usage
    
    
    trac-inherit-setup
         boottrap the global trac conf $(trac-inheritpath) 
    
    trac-upgrade
         edit the trac.ini from all the instances making changes to 
         work with trac 0.11
 
         this is poorly named trac-configure-all would be more appropriate
 
 
 
    trac-setbanner  <path-to-banner>    
          Path defaults to \$(trac-bannerpath) : $(trac-bannerpath)
          
          Copies the banner to the environment htdocs directory, 
          and configures trac.ini to use it via the header_logo block
          using a "site" url prefix which refers to the environment 
          htdocs dir.
          
          The default banner is  common/trac_banner.png  236x73 pixels
 
          Target non-default instance with :
              TRAC_INSTANCE=dybsvn trac-setbanner
 
   
   
    trac-intertrac-conf
         see InterTrac to check it worked
             SUDO=sudo TRAC_INSTANCE=.. trac-intertrac-conf
         targetting ".." will put the config in the inherited ini file, used by 
         all instances on the server
    
    trac-timeline-conf
        see TracIni for the options  
   
       SUDO=sudo TRAC_INSTANCE=dybsvn trac-timeline-conf
   
  
    trac-notification-conf <email>   
         see wiki:TracNotification
         eg :   
               trac-notification-conf theta13-offline@lists.lbl.gov offline
           the 2nd argument is a username/email to be used for bitten notification,
           of failed builds 

          includes conf used by bittennotify, if that is enabled :
             notification:notify_on_failed_build:true
             notification:notify_on_successful_build:true
 
 
 
    trac-db  <name>     defaults to TRAC_INSTANCE : $TRAC_INSTANCVE
          echo command for direct access to the trac database with sqlite3 ... it is
          not advisable to do this on a production database, i do it on recoverd backup 
          databases on non-production machines
  
 
EOU

}

tscript-(){           . $ENV_HOME/trac/script/tscript.bash  && tscript-env  $* ; }
tracinter-(){         . $ENV_HOME/trac/tracinter.bash  && tracinter-env  $* ; }
tracbuild-(){         . $ENV_HOME/trac/tracbuild.bash  && tracbuild-env  $* ; }
tracperm-(){          . $ENV_HOME/trac/tracperm.bash   && tracperm-env   $* ; }
traccomp-(){          . $ENV_HOME/trac/traccomp.bash   && traccomp-env   $* ; }
nuwacomp-(){          . $ENV_HOME/trac/nuwacomp.bash   && nuwacomp-env   $* ; }
autocomp-(){          . $ENV_HOME/trac/autocomp/autocomp.bash   && autocomp-env   $* ; }


bitextra-(){          . $ENV_HOME/trac/package/bitextra.bash  && bitextra-env  $* ; }
tractags-(){          . $ENV_HOME/trac/package/tractags.bash  && tractags-env $* ; }
tracnav-(){           . $ENV_HOME/trac/package/tracnav.bash   && tracnav-env  $* ; }
tractoc-(){           . $ENV_HOME/trac/package/tractoc.bash   && tractoc-env  $* ; }
accountmanager-(){    . $ENV_HOME/trac/package/accountmanager.bash    && accountmanager-env   $* ; }
bitten-(){            . $ENV_HOME/trac/package/bitten.bash    && bitten-env   $* ; }
tractrac-(){          . $ENV_HOME/trac/package/tractrac.bash  && tractrac-env $* ; }
genshi-(){            . $ENV_HOME/trac/package/genshi.bash    && genshi-env   $* ; }
trac2mediawiki-(){    . $ENV_HOME/trac/package/trac2mediawiki.bash    && trac2mediawiki-env   $* ; }
trac2latex-(){        . $ENV_HOME/trac/package/trac2latex.bash    && trac2latex-env   $* ; }


silvercity-(){        . $ENV_HOME/trac/package/silvercity.bash && silvercity-env   $* ; }
pygments-(){          . $ENV_HOME/trac/package/pygments.bash   && pygments-env   $* ; }
docutils-(){          . $ENV_HOME/trac/package/docutils.bash   && docutils-env   $* ; }
textile-(){           . $ENV_HOME/trac/package/textile.bash    && textile-env   $* ; }

bittennotify-(){      . $ENV_HOME/trac/package/bittennotify.bash && bittennotify-env   $* ; }
tracreposearch-(){    . $ENV_HOME/trac/package/tracreposearch.bash && tracreposearch-env   $* ; }


trac-instance(){
    [ -n "$TRAC_INSTANCE_OVERRIDE" ] && echo $TRAC_INSTANCE_OVERRIDE && return 1  
    case ${1:-$NODE_TAG} in
     G) echo dybsvn   ;;
     H) echo env      ;;
     P) echo env      ;;
     C) echo env      ;;
     *) echo dybsvn   ;;
   esac
}


trac-version(){
   case ${1:-$NODE_TAG} in
  ## G) echo 0.11rc1 ;;
     G) echo 0.11 ;;
     H) echo 0.10.4  ;;
     P) echo 0.11    ;;
     C) echo 0.11    ;;
     *) echo 0.11    ;;
   esac
}


trac-baseurl(){
   case ${1:-$NODE_TAG} in 
      G) echo http://localhost ;;
      H) echo http://dayabay.phys.ntu.edu.tw ;;
      P) echo http://grid1.phys.ntu.edu.tw:8080 ;;
      C) echo http://dayabay.phys.ntu.edu.tw ;;
      *) echo http://localhost ;;
   esac
}


trac-url(){
   echo $(trac-baseurl)/tracs/${1:-$TRAC_INSTANCE}
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
trac-repopath(){ 
   local name=${1:-$TRAC_INSTANCE}
   
   ## special case for recoverd dybsvn on G 
   if [ "$NODE_TAG" == "G" -a "$name" == "dybsvn" ]; then
      utag="XX"
   else
      utag=$NODE_TAG   
   fi
   svn-
   echo $SCM_FOLD/$(svn-repo-dirname $utag)/$name 
}
trac-logpath(){ echo $(trac-envpath $*)/log/trac.log ; }
trac-inipath(){ echo $(trac-envpath $*)/conf/trac.ini ; }
trac-pkgpath(){ echo $ENV_HOME/trac/package ; }

trac-inheritpath(){ echo $SCM_FOLD/conf/trac.ini ; }  


trac-administrator(){
   case $NODE_TAG in 
      XX) echo tianxc ;;
       *) echo blyth ;;
   esac
}


trac-tail(){ tail -n 50 -f $(trac-logpath $*) ; }
trac-log(){  cd $(dirname $(trac-logpath $*)) ; ls -l  ;}
trac-inicat(){  cat $(trac-inipath $*) ; }
trac-inhcat(){  cat $(trac-inheritpath $*) ; }
trac-vi(){     $SUDO vi $(trac-inipath $*) ; }

trac-admin-(){   $SUDO trac-admin $(trac-envpath) $* ; }
trac-configure(){ trac-edit-ini $(trac-inipath) $*   ; }
trac-edit-ini(){

   local path=$1
   local user=$TRAC_USER
   shift
 
   $SUDO perl $ENV_HOME/base/ini-edit.pl $path $*  
   [ -n "$SUDO" ] && $SUDO chown $user:$user $path
}

trac-db(){
   echo sqlite3 $(trac-envpath $*)/db/trac.db
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
    
    if [ ! -d $dir ]; then
        echo $msg creating dir $dir for global inherited conf 
        $SUDO mkdir -p $dir 
        [ -n "$SUDO" ] && $SUDO chown $user:$user $dir
    fi
       
    if [ ! -f $inherit ]; then 
        echo $msg bootstraping global config $inherit
        trac-inherit 
    fi
}


trac-inherit(){

   local live=$(trac-inheritpath)
   local path=${1:-$live}
   local tmp=/tmp/env/${FUNCNAME/-*/} && mkdir -p $tmp
   local name=$(basename $path)
   local user=$TRAC_USER
   
   trac-inherit- > $tmp/$name
   
   if [ "$path" == "$live" ]; then
      $SUDO cp $tmp/$name $live 
      [ -n "$SUDO" ] && $SUDO chown $user:$user $live
   fi

}


trac-inherit-(){
 
   cat << EOI

##
## do not edit this file ... 
##   it is sourced from trac/trac.bash::trac-inherit- in env repository
##    $BASH_SOURCE 
##
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
#trac2latex.* = enabled
trac2mediawiki.* = enabled
tracnav.* = enabled
#tracreposearch.* = enabled
tractoc.* = enabled
webadmin.* = enabled
xslt.* = enabled



   
   
EOI

}



trac-configure-all(){

    local msg="=== $FUNCNAME :"
    echo $msg this is re-named from upgrade as configure-all is more appropriate

    for name in $(trac-instances)
    do
       trac-comment $name "tracrpc.* = enabled"
       trac-comment $name "default_handler = TagsWikiModule"
       trac-comment $name "trac.wiki.web_ui.wikimodule = disabled"
       
       TRAC_INSTANCE=$name trac-configure  $(trac-triplets $name)    $(trac-enscript $name)   
    done
}


trac-admin-upgrade(){

    for name in $(trac-instances)
    do 
        TRAC_INSTANCE=$name trac-admin- upgrade
        TRAC_INSTANCE=$name trac-admin- wiki upgrade
    done

}


trac-admin-all-(){
   local msg="=== $FUNCNAME :"
   for name in $(trac-instances)
   do 
        echo $msg $name
        TRAC_INSTANCE=$name trac-admin- $*
   done
}






trac-comment(){

   local msg="=== $FUNCNAME :"
   local name=${1:-$TRAC_INSTANCE}
   local skip="$2"
   local path=$(trac-inipath $name)
   local user=$TRAC_USER
   
   echo $msg commenting "$skip" from $path user $user
   $SUDO perl -pi -e "s,^($skip),#\$1 ## removed by $BASH_SOURCE::$FUNCNAME ,  " $path
   [ -n "$SUDO" ] && $SUDO chown $user:$user $path

}


trac-enscript(){
   enscript-
   local path=$(enscript-dir)/bin/enscript
   if [ -x "$path" ]; then
      echo mimeviewer:enscript_path:$path
   fi
}


trac-logging(){
    
    local mbytes=1
    local count=10
    
    trac-configure $(cat << EON 
        logging:log_maxsize:$mbytes
        logging:log_maxcount:$count
EON)


}


trac-triplets(){

   local name=${1:-$TRAC_INSTANCE}
   svn-
   local authz=$(svn-authzpath)
   local users=$(svn-userspath)
   local url=$(TRAC_INSTANCE=$name trac-url)
   local repo=$(TRAC_INSTANCE=$name trac-repopath)
   local inherit=$(trac-inheritpath)
   
   
   cat << EOT
      inherit:file:$inherit
      trac:authz_file:$authz
      trac:repository_dir:$repo
      account-manager:password_file:$users
      trac:base_url:$url
      header_logo:link:$url
      project:url:$url
      ticket:restrict_owner:true
EOT

}



trac-bannerpath(){
   local path=$ENV_HOME/logo/theta13_offline_software.png
   [ -f $path ] && echo $path || echo -n
} 


trac-setbanner(){

   local msg="=== $FUNCNAME :" 
   local path=${1:-$(trac-bannerpath)}
   [ ! -f "$path" ] && echo $msg no such path $path && return 1
   
   local cmd="$SUDO cp -f $path $(trac-envpath)/htdocs/"
   echo $msg $cmd
   eval $cmd
   
   
   local name=$(basename $path)
   local banner="site/$name"
   
    
   
   trac-configure header_logo:src:$banner
  
}




trac-intertrac-conf(){
   tracinter-
   trac-configure $(tracinter-triplets)
}

trac-timeline-conf(){
  trac-configure $(cat << EOC
     timeline:ticket_show_details:true 
     timeline:changeset_show_files:-1
     timeline:changeset_long_messages:true
EOC)
}


trac-bitten-exclude(){
  local x=$(cat << EOX | tr "\n" "," 
dybspade
people
groups
vendor
NuWa
installation/branches
dybgaudi/branches
dybgaudi/tags
tutorial/branches
relax/branches
lhcb/branches
lcgcmt/branches
ldm/branches
EOX)
 echo ${x:0:$((${#x}-1))}   ## just to remove the trailing comma
}

trac-bitten-conf(){
  trac-configure $(cat << EOC
     bitten:dybinst.exclude_paths:$(trac-bitten-exclude)
EOC)
}

trac-notification-conf(){

  local email=$1
  local bmail=${2:-$email}
  local always=""
  local build=""
  [ -n "$email" ] && always=notification:smtp_always_cc:$email && build=notification:bitten_build_cc:$bmail

  trac-configure $(cat << EOC
     notification:smtp_default_domain:localhost 
     notification:smtp_enabled:true 
     notification:use_public_cc:true
     notification:always_notify_owner:true
     notification:always_notify_reporter:true
     notification:always_notify_updater:true
     $always
     components:bittennotify.*:enabled
     notification:notify_on_failed_build:true
     notification:notify_on_successful_build:false
     $build
EOC)

}






