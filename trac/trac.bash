trac-src(){    echo trac/trac.bash ; }
trac-source(){ echo ${BASH_SOURCE:-$(env-home)/$(trac-src)} ; }
trac-vi(){     vi $(trac-source) ; }
trac-usage(){
cat << EOU


   == belle1 : try epel 5 trac 0.10.5 == 



   [blyth@belle1 ~]$ sudo yum --enablerepo=epel install trac
   Loaded plugins: kernel-module
   Setting up Install Process
   Package trac-0.10.5-3.el5.noarch already installed and latest version
   Nothing to do
   [blyth@belle1 ~]$ 
   [blyth@belle1 ~]$ 
   [blyth@belle1 ~]$ 
   [blyth@belle1 ~]$ rpm -ql trac
   /etc/httpd/conf.d/trac.conf
   /usr/bin/trac-admin
   /usr/lib/python2.4/site-packages/trac
   /usr/lib/python2.4/site-packages/trac/About.py
   /usr/lib/python2.4/site-packages/trac/About.pyc
   /usr/lib/python2.4/site-packages/trac/About.pyo
   ..
/usr/lib/python2.4/site-packages/trac/wiki/web_ui.pyo
/usr/sbin/tracd
/usr/share/doc/trac-0.10.5
/usr/share/doc/trac-0.10.5/AUTHORS
/usr/share/doc/trac-0.10.5/COPYING
/usr/share/doc/trac-0.10.5/ChangeLog
/usr/share/doc/trac-0.10.5/INSTALL
/usr/share/doc/trac-0.10.5/README
/usr/share/doc/trac-0.10.5/README.tracd
/usr/share/doc/trac-0.10.5/RELEASE
/usr/share/doc/trac-0.10.5/THANKS
/usr/share/doc/trac-0.10.5/UPGRADE
/usr/share/doc/trac-0.10.5/contrib
/usr/share/doc/trac-0.10.5/contrib/README
/usr/share/doc/trac-0.10.5/contrib/bugzilla2trac.py
/usr/share/doc/trac-0.10.5/contrib/emailfilter.py
/usr/share/doc/trac-0.10.5/contrib/htdigest.py
/usr/share/doc/trac-0.10.5/contrib/migrateticketmodel.py
/usr/share/doc/trac-0.10.5/contrib/sourceforge2trac.py
/usr/share/doc/trac-0.10.5/contrib/trac-post-commit-hook
/usr/share/doc/trac-0.10.5/contrib/trac-post-commit-hook.cmd
/usr/share/doc/trac-0.10.5/contrib/trac-pre-commit-hook
/usr/share/man/man1/trac-admin.1.gz
/usr/share/man/man8/tracd.8.gz
/usr/share/trac
/usr/share/trac/conf
/usr/share/trac/htdocs
/usr/share/trac/htdocs/README
/usr/share/trac/htdocs/asc.png
/usr/share/trac/htdocs/attachment.png
/usr/share/trac/htdocs/changeset.png
/usr/share/trac/htdocs/closedticket.png
/usr/share/trac/htdocs/css
...
/usr/share/trac/wiki-macros/TracGuideToc.pyc
/usr/share/trac/wiki-macros/TracGuideToc.pyo
/var/www/cgi-bin/trac.cgi
/var/www/cgi-bin/trac.fcgi




   uh oh, epel aint so smooth:


[blyth@belle1 trac-0.10.5]$ cat  /etc/httpd/conf.d/trac.conf
# Replace all occurrences of /srv/trac with your trac root below
# and uncomment the respective SetEnv and PythonOption directives.
<LocationMatch /cgi-bin/trac\.f?cgi>
    #SetEnv TRAC_ENV /srv/trac
    </LocationMatch>
    <IfModule mod_python.c>
    <Location /cgi-bin/trac.cgi>
        SetHandler mod_python
	    PythonHandler trac.web.modpython_frontend
	        #PythonOption TracEnv /srv/trac
		</Location>
		</IfModule>












   Trac Customizations to adopt ?
       http://webmail.inoi.fi/open/trac/eunuchs
          Nice slim trac instance navigator at top of page, with "select-trac" CSS 

   -------
       Putting trac inside a virtualenv has non-first glance advantages :
          * avoid sudo shenanigans   

   --------

   This attempts to automate Trac installation as far as possible
   documentation of the manual installation of 0.11b1 is at 
   
      http://dayabay.phys.ntu.edu.tw/tracs/env/wiki/LeopardTrac

   Considerations...
      0) all packages including package to be installed by a standard procedure
      1) do i need to kickstart the tracitory from a prior backup 


    NODE_TAG      : $NODE_TAG

    trac-version  : $(trac-version)
    trac-instance : $(trac-instance)    the default instance for the node    
    trac-site     : $(trac-site)    
         ihep .. for dybsvn, dybaux and toysvn instances otherwise ntu , 
         used for distinguiishing IHEP/NTU layout differences 
    
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
    trac-edit      <name>   
           for name of ".." this edits the common inherited config  
           
    trac-rename <oldname> <newname>
           renames the instance and redoes the trac-configure-instance in the newly named one 
           ... this should only be done in parallel to the svn-rename preferably by using scm-rename
      
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
           
           
    trac-admin-- <args> 
            invokation of trac-admin- <args> from within "sudo bash -c" with the
            environment hooked up   
           
    trac--  <args>
        for sudo bash usage of "trac-*" functions, eg 
            trac-- TRAC_INSTANCE=newtest trac-admin- permission list
           
                             
    trac-configure  <block:qty:valu> ... 
           applies edits to  trac.ini by means of triplet arguments
   
    trac-configure-instance <name>
           applies a standard set of config triplets to the named 
           instance         
           ... may need to upgrade the environment after doing this 
 
 
    trac-configure-all   (formerly poorly named as "trac-upgrade")
           invokes trac-configure-instance for all instances
           edits the trac.ini from all the instances making changes to 
           work with trac 0.11 and the standard set of plugins
 
    
 

    trac-logging 
          this relies on local patch mods, see 
            http://dayabay.phys.ntu.edu.tw/tracs/env/wiki/TracTweaking#traclogs    
 
 
 
 
    $(type trac-prepare)
    
    
    
    trac-admin-upgrade
         do trac-admin "upgrade" and "wiki upgrade" to 
         bring 0.10.4 instances up to 0.11 usage
    
    
    trac-inherit-setup
         boottrap the global trac conf $(trac-inheritpath) 
    
    
    trac-placebanner  <path-to-banner>    
          Path defaults to \$(trac-bannerpath) : $(trac-bannerpath)
          Copies the banner to the environment htdocs directory, 

          Configuring trac.ini to use it is done by trac-configure
          via the trac-banner-triplet
          
    trac-banner-triplet
          supplies the triplet for the banner based on file existence
          when a non-default banner is present supplies a "site" 
          url prefix which refers to the environment htdocs dir.
 

  
         
          The default banner is  common/trac_banner.png  236x73 pixels
 
          Target non-default instance with :
              SUDO=sudo TRAC_INSTANCE=dybsvn   trac-placebanner
              SUDO=sudo TRAC_INSTANCE=env      trac-placebanner
              SUDO=sudo TRAC_INSTANCE=aberdeen trac-placebanner
   
   
   
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
tmacros-(){           . $ENV_HOME/trac/macros/macros.bash   && tmacros-env  $* ; }
tracinter-(){         . $ENV_HOME/trac/tracinter.bash  && tracinter-env  $* ; }
tracbuild-(){         . $ENV_HOME/trac/tracbuild.bash  && tracbuild-env  $* ; }
tracperm-(){          . $ENV_HOME/trac/tracperm.bash   && tracperm-env   $* ; }
traccomp-(){          . $ENV_HOME/trac/traccomp.bash   && traccomp-env   $* ; }
tracinit-(){          . $ENV_HOME/trac/tracinit.bash   && tracinit-env   $* ; }
nuwacomp-(){          . $ENV_HOME/trac/nuwacomp.bash   && nuwacomp-env   $* ; }
autocomp-(){          . $ENV_HOME/trac/autocomp/autocomp.bash   && autocomp-env   $* ; }


navadd-(){            . $ENV_HOME/trac/package/navadd.bash    && navadd-env   $* ; }
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
fullblog-(){          . $ENV_HOME/trac/package/fullblog.bash    && fullblog-env   $* ; }


silvercity-(){        . $ENV_HOME/trac/package/silvercity.bash && silvercity-env   $* ; }
pygments-(){          . $ENV_HOME/trac/package/pygments.bash   && pygments-env   $* ; }
docutils-(){          . $ENV_HOME/trac/package/docutils.bash   && docutils-env   $* ; }
textile-(){           . $ENV_HOME/trac/package/textile.bash    && textile-env   $* ; }

ofc2dz-(){            . $(env-home)/trac/ofc2dz.bash && ofc2dz-env $* ; }



bittennotify-(){      . $ENV_HOME/trac/package/bittennotify.bash && bittennotify-env   $* ; }
tracreposearch-(){    . $ENV_HOME/trac/package/tracreposearch.bash && tracreposearch-env   $* ; }




trac--(){
   sudo bash -c "export ENV_HOME=$ENV_HOME ; . $ENV_HOME/env.bash ; env- ; trac- ; $* "
}


trac-home(){
   ## home of the pertinent working copy 
   local name=${1:-$TRAC_INSTANCE}
   local f="$name-home"
   $f 
}

trac-instance(){
    ## the override should be set in .bash_profile for a temporary change to the default instance
    [ -n "$TRAC_INSTANCE_OVERRIDE" ] && echo $TRAC_INSTANCE_OVERRIDE && return 1  
    case ${1:-$NODE_TAG} in
              G) echo workflow ;;
     C|C2|N|P|H) echo env      ;;
       XX|YY|ZZ) echo dybsvn   ;;
              *) echo env      ;;
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


trac-baseurl-deprecated-use-env-localserver(){
   case ${1:-$NODE_TAG} in 
      G) echo http://localhost ;;
      H) echo http://dayabay.phys.ntu.edu.tw ;;
      P) echo http://grid1.phys.ntu.edu.tw:8080 ;;
      C) echo http://dayabay.phys.ntu.edu.tw ;;
      *) echo http://localhost ;;
   esac
}


trac-find(){
   python-
   find $(python-site)/Trac-0.11-py2.5.egg/trac -name '*.py' -exec grep -H $1 {} \;
}



trac-url(){
   local name=${1:-$TRAC_INSTANCE}
   case $name in 
     workflow) echo http://localhost/tracs/$name ;;
            *) echo $(env-localserver)/tracs/${1:-$TRAC_INSTANCE} ;;
   esac
} 
 


trac-build--(){     screen bash -lc "trac-;trac-build" ; }
trac-build(){


   ## create the base folders
   local-
   local-initialize

   tracpreq-
   tracpreq-again

   tracbuild-
   tracbuild-auto 

   # do this in local-initialize 
   #trac-inherit-setup

}


trac-check(){

   local msg="=== $FUNCNAME :"
   configobj-
   ! configobj-check && echo $msg configobj-check FAILED ... SLEEPING && sleep 100000000000000 

}



trac-env(){
   elocal-
   package-
   
   export TRAC_INSTANCE=$(trac-instance)
   export TRAC_VERSION=$(trac-version)
   export TRAC_USER=$(trac-user)  ## DEPRECATED DO NOT USE 
  
   # these settings ?were? used by svn-apache-* for apache2 config 
   # apache-
   # export TRAC_APACHE2_CONF=$APACHE2_LOCAL/trac.conf 
   #
 
   # when packages need to be installed in a particular order arrange
   # them here ... the rest will be added to the end in alphabetical order
   #
   export TRAC_NAMES_BASE="genshi tractrac bitten" 
   
   
   
}


trac-localserver(){ 
  env-localserver
}

trac-user(){
   apache-
   echo $(apache-user)
}
trac-group(){
   apache-
   echo $(apache-group)
}

trac-site(){
  case ${1:-$TRAC_INSTANCE} in
  dybsvn|toysvn|dybaux) echo ihep ;;
               mdybsvn) echo ntu  ;;
                     *) echo ntu  ;;
  esac
}

trac-major(){   echo ${TRAC_VERSION:0:4} ; }
trac-envpath(){ echo $SCM_FOLD/tracs/${1:-$TRAC_INSTANCE} ; }
trac-repopath(){ 
   local name=${1:-$TRAC_INSTANCE}
   ## special case for recovered instances from IHEP  
   local site=$(trac-site $name)
   svn-
   echo $SCM_FOLD/$(svn-repo-dirname-forsite $site)/$name 
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


trac-iheplog-vi(){
   vi $(trac-iheplog-dir)/trac.log
}
trac-iheplog-cd(){
   cd $(trac-iheplog-dir)
}
trac-iheplog-dir(){
   echo /tmp/$USER/env/trac/$FUNCNAME 
}
trac-iheplog-get(){

   local msg="=== $FUNCNAME :"
   #local url=http://dayabay.ihep.ac.cn/log/trac.log
   local url=http://dayabay.ihep.ac.cn/log/trac/log/trac.log
   local dir=$(trac-iheplog-dir)
   mkdir -p $dir
   cd $dir
   private-
   local cmd="curl -u $(private-val IHEP_TRACLOG_USER):$(private-val IHEP_TRACLOG_PASS) -O $url"
   echo $msg $cmd
   eval $cmd

}

trac-tail(){ tail -n 50 -f $(trac-logpath $*) ; }
trac-log(){  cd $(dirname $(trac-logpath $*)) ; ls -l  ;}
trac-lvi(){  vi $(trac-logpath $*) ;}
trac-inicat(){  cat $(trac-inipath $*) ; }
trac-inhcat(){  cat $(trac-inheritpath $*) ; }
trac-edit(){    $SUDO vi $(trac-inipath $*) ; }

trac-logname(){ echo trac.log ; }

trac-admin--(){
   sudo bash -c "export ENV_HOME=$ENV_HOME ; . $ENV_HOME/env.bash ; env- ; trac- ; TRAC_INSTANCE=$TRAC_INSTANCE trac-admin- $* "
  ## huh this did not change ownership of the trac.log ??
}


trac-admin-sqlite-check-(){
  python $(env-home)/trac/sqlite-version-check.py
}
trac-admin-sqlite-check(){
   ## NB the sqlite3 version that matters is the one installed into python site-packages 
   local v=$($FUNCNAME-)
   case $v in
     "sqlite_version_string:3.1.2 have_pysqlite:1") echo $msg ABORT this combination has resulted in memory exhaution within seconds ...  see $(env-wikiurl)/TracSQLiteMemoryExhaustion && return 1 ;; 
     "sqlite_version_string:3.3.16 have_pysqlite:2") echo $msg OK $v   ;;
     "sqlite_version_string:3.3.6 have_pysqlite:2") echo $msg MAYBE OK $v ON N WITH SYSTEM python ;;
     "sqlite_version_string:3.4.0 have_pysqlite:2" ) echo $msg OK $v ... apples sqlite3 ;;
     "sqlite_version_string:3.7.5 have_pysqlite:2" ) echo $msg UNKNOWN  ;;
                                                  *) echo $msg ABORT non-supported sqlite/pysqlite version $v ... see $(env-wikiurl)/TracSQLiteMemoryExhaustion && return 1 ;;  
   esac
}

trac-admin-(){   

   local msg="=== $FUNCNAME :"

   python-
   sqlite-

   echo $msg trac-admin : $(which trac-admin)
   echo $msg python     : $(which python)
   echo $msg LLP        : $LD_LIBRARY_PATH | tr ":" "\n"

   trac-admin-sqlite-check

   local rc=$?
  
   [ "$rc" != "0" ] && echo env-abort && env-abort 

   local cmd="trac-admin $(trac-envpath) $*"
   echo $msg $cmd
   eval $cmd 
}


trac-configure(){ trac-edit-ini $(trac-inipath) $*   ; }


trac-edit-ini(){
   local msg="=== $FUNCNAME :"
   local path=$1
   shift
   
   local user=$(trac-user)
   local tmp=/tmp/env/trac/$FUNCNAME && mkdir -p $tmp
   local tpath=$tmp/$(basename $path)
   
   ## edit a temporary copy of the ini file
   local cmd="cp $path $tpath "
   eval $cmd
   python $ENV_HOME/base/ini.py $tpath $*  
   
   
   local dmd="diff $path $tpath" 
   echo $msg $dmd
   eval $dmd

   [ "$?" == "0" ] && echo $msg no differences ... skipping && return 0

   if [ -n "$TRAC_CONFIRM" ]; then
      local ans
      read -p "$msg enter YES to confirm this change " ans
      [ "$ans" != "YES" ] && echo $msg skipped && return 1
   fi 

   
   
   $SUDO cp $tpath $path 
   [ "$user" != "$USER" ] &&  $SUDO chown $user:$user $path

}


trac-edit-ini-deprecated(){

   local path=$1
   local user=$(trac-user)
   shift
 
   $SUDO perl $ENV_HOME/base/ini-edit.pl $path $*  
   [ -n "$SUDO" ] && $SUDO chown $user:$user $path
}

trac-db(){
   echo sqlite3 $(trac-envpath $*)/db/trac.db
}

trac-rename(){

   local msg="=== $FUNCNAME :" 
   local oldname=$1
   local newname=$2
   [ -z "$oldname" ]     && echo $msg an existing instance name must be provided as the one to rename && return 1
   [ -z "$newname" ]     && echo $msg an non-existing instance name must be provided as the newname   && return 1
   ! trac-exists $oldname  && echo $msg ABORT an no instance with name \"$oldname\" exists  && return  1 
   trac-exists $newname    && echo $msg ABORT an instance with name \"$newname\" exists already && return  1 
     
   local iwd=$PWD
   local oldenv=$(trac-envpath $oldname);
   local dir=$(dirname $oldenv);
   cd $dir;
   local cmd="sudo mv $oldname $newname";
   echo $msg $cmd;
   eval $cmd;

   trac-configure-instance $newname 

   echo $msg resyncing the instance with the repository ... as repository_dir has changed ... avoiding the yellow banner;
   TRAC_INSTANCE=$newname trac-admin-- resync

   echo $msg ensure everything in the newenv path is accessible to apache ... have observed trac.log converting to root somehow
   apache-
   sudo find $(trac-envpath $newname) -group root -exec chown $(apache-user):$(apache-group) {} \; 


   cd $iwd
}



trac-create(){
   local msg="=== $FUNCNAME :" 
   local name=$1
   [ -z "$name" ]     && echo $msg an instance name must be provided && return 1
   trac-exists $name  && echo $msg ABORT an instance with name \"$name\" exists already && return  1 
   
   tracinit-
   tracinit--  tracinit-prepare $name
  
}

trac-exists(){
   local name=$1
   local inst
   for inst in $(trac-instances) ; do
      [ "$name" == "$inst" ] && return 0
   done
   return 1
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


trac-wipe(){

   local iwd=$PWD
   local msg="=== $FUNCNAME :"
   local name=$1
   [ -z $SCM_FOLD ] && echo $msg ABORT no SCM_FOLD && return 1
   
   ! trac-exists $name && echo $msg ABORT no such instance exists with name \"$name\" && return 1
   
   local inst=$(trac-envpath $name)
   local dir=$(dirname $inst) 
   
   cd $dir
   [ ! -d "$name" ] && echo $msg ABORT instance \"$name\" does not exist && return 1
   
   local answer
   read -p "$msg are you sure you want to wipe the instance \"$name\" from $dir ? YES to proceed " answer
   [ "$answer" != "YES" ] && echo $msg skipping && return 1
   [ ${#name} -lt 3 ]  && echo $msg name $name is too short not proceeding && return 1
   
   local cmd="$SUDO rm -rf \"$name\""
   echo $msg $cmd 
   eval $cmd

   cd $iwd
}









trac-prepare(){

   trac-inherit-setup
   trac-configure-all

}

trac-prepare-instance(){
   local msg="=== $FUNCNAME :"
   local name=$1
   [ -z "$name" ] && echo $msg the name of an instance must be provided && return 1
   
   trac-configure-instance $name
   TRAC_INSTANCE=$name trac-admin- upgrade

}

trac-inherit-setup(){

   [ "$(trac-major)" != "0.11" ] && echo $msg this is only relevant to 0.11 && return 1
   
    local msg="=== $FUNCNAME :"
    local inherit=$(trac-inheritpath)
    local dir=$(dirname $inherit)
    local user=$(trac-user)
    local group=$(trac-group)
    
    if [ ! -d $dir ]; then
        echo $msg creating dir $dir for global inherited conf 
        $SUDO mkdir -p $dir 
        [ -n "$SUDO" ] && $SUDO chown $user:$group $dir
    fi
        
    trac-inherit 
}


trac-inherit-(){


   cat << EOH
#
#  this was created by $(trac-source)::$FUNCNAME 
#
EOH

    cat $ENV_HOME/trac/common.ini

    trac-enscript-inherited

    tracinter-
    tracinter-ini

}


trac-inherit(){

   local msg="=== $FUNCNAME :"
   local live=$(trac-inheritpath)
   local path=${1:-$live}
   local tmp=/tmp/env/${FUNCNAME/-*/} && mkdir -p $tmp
   local tpath=$tmp/$(basename $path)
   
   apache-
   local user=$(apache-user)
   
   trac-inherit- > $tpath
 
   local dmd="diff $live $tpath"
   $dmd > /dev/null
   local rc=$?
 
   if [ "$rc" == "0" ]; then
      echo $msg proposed inherit config $tpath is the same as the current one $live
   else
      echo $msg a changed inherit config $tpath is proposed ...
      echo $msg $dmd
      $dmd
   
      local ans
      read -p "install the new config ? Enter YES to do so :" ans 
      if [ "$ans" == "YES" ]; then
         $SUDO cp $tpath $live 
       
         if [ -n "$SUDO" ]; then 
            local cmd="$SUDO chown $user:$user $live"
            echo $cmd
            eval $cmd
         fi
      else
         echo $msg skipping new config 
      fi
      
      
   fi

}


trac-configure-instance(){

  local msg="=== $FUNCNAME :"
  local name=${1:-$TRAC_INSTANCE}

  [ -z "$name" ] && echo $msg the name of the instance is required && return 1
  [ ! -d "$(trac-envpath $name)" ] && echo $msg ABORT no such trac environment $(trac-envpath $name) && return 1
 
  local inherit=$(trac-inheritpath)
  [ ! -f "$inherit" ] && echo $msg ABORT ... sleeping ... there is no trac-inheritpath : $inherit ... do trac-inherit-setup elsewhere then ctrl-C out of the sleep to continue && sleep 100000000
 
  ## this is needed to do copies of logos ... into htdocs
  SUDO=sudo TRAC_INSTANCE=$name   trac-placebanner 
  TRAC_INSTANCE=$name trac-configure  $(trac-triplets $name)  

}






trac-delete-triplets(){

 # These should all be configured in the inherited common.ini ... rather than the individual trac.ini

 # trac-comment $name "tracrpc.* = enabled"
 # trac-comment $name "default_handler = TagsWikiModule"
 # trac-comment $name "trac.wiki.web_ui.wikimodule = disabled"      housed in common.ini not needed here ?
 # trac-comment $name "enscript_path = .*"

 #  not there anymore ?
 # trac:default_handler:TagsWikiModule@DELETE

  cat << EOT
      components:tracrpc.*:enabled@DELETE
      components:trac.wiki.web_ui.wikimodule:disabled@DELETE
      mimeviewer:enscript_path:@DELETE
EOT
}



trac-configure-all(){

    local msg="=== $FUNCNAME :"
    echo $msg this is re-named from upgrade as configure-all is more appropriate

    for name in $(trac-instances)
    do
       trac-configure-instance $name
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
   local user=$(trac-user)
   local group=$(trac-group)
   
   echo $msg commenting "$skip" from $path user $user
   $SUDO perl -pi -e "s,^($skip),#\$1 ## removed by $BASH_SOURCE::$FUNCNAME ,  " $path
   [ -n "$SUDO" ] && $SUDO chown $user:$group $path

}


trac-enscript-deprecated(){
   enscript-
   local path=$(enscript-dir)/bin/enscript
   if [ -x "$path" ]; then
      echo mimeviewer:enscript_path:$path
   fi
}

trac-enscript-inherited(){
  enscript-
  local path=$(enscript-dir)/bin/enscript
  [ ! -x "$path" ] && echo "## $FUNCNAME .. WARNING : NO enscript AT $path " && return 0
  
  cat << EOI

## constructed by : $FUNCNAME 
[mimeviewer]
enscript_path = $path

EOI

}


trac-logging(){
    
    local mbytes=1
    local count=10
 
    trac-configure $(cat << EON 
        logging:log_maxsize:$mbytes
        logging:log_maxcount:$count
EON)
}


trac-defaulthandler-triplets-(){
   local name=${1:-$TRAC_INSTANCE}
   case $name in
          dybsvn|env|workflow) echo Wiki ;;
                       dybaux) echo Wiki ;;
                            *) echo Wiki     ;; 
   esac
}
trac-defaulthandler-triplets(){ echo trac:default_handler:$($FUNCNAME- $*)Module ;  }



trac-mainnav-triplets-(){
   local name=${1:-$TRAC_INSTANCE}
   case $name in 
      dybaux) echo -roadmap -query -newticket -tickets ;;
           *) echo -n ;;
   esac   
}
trac-mainnav-triplets(){
  local tab
  $FUNCNAME- $* | tr " " "\n" | while read tab ; do
     case ${tab:0:1} in
        +) echo mainnav:${tab:1}:enabled ;;
        -) echo mainnav:${tab:1}:disabled ;;
     esac
  done
}



trac-defaultquery-triplets(){
   local name=${1:-$TRAC_INSTANCE}
   case $name in
       dybsvn)  echo 'query:default_query:status!=closed&owner=$USER|offline'  ;;
 env|workflow)  echo 'query:default_query:status!=closed&owner=$USER|admin&group=component'    ;;
            *)  echo 'query:default_query:status!=closed&owner=$USER'          ;;
   esac 
}

trac-triplets(){
   local name=${1:-$TRAC_INSTANCE}
   svn-
   local authz=$(svn-authzpath)
   local users=$(svn-userspath)
   local url=$(TRAC_INSTANCE=$name trac-url)
   local repo=$(TRAC_INSTANCE=$name trac-repopath)
   local inherit=$(trac-inheritpath)
   
   navadd-
   
   cat << EOT
      inherit:file:$inherit
      trac:authz_file:$authz
      trac:authz_module_name:$name
      trac:repository_dir:$repo
      account-manager:password_file:$users
      account-manager:password_store:HtPasswdStore
      trac:base_url:$url
      header_logo:link:$url
      header_logo:alt:
      $(TRAC_INSTANCE=$name trac-banner-triplet)
      logging::
      wiki::
      project:url:$url
      project:name:$name
      project:descr:$name
      tags:ignore_closed:false
      tags:ticket_fields:keywords
      ticket:restrict_owner:true
$(trac-delete-triplets $name)
$(navadd-triplets query Query /tracs/$name/query)
      $(trac-defaulthandler-triplets $name)
      $(trac-defaultquery-triplets $name)
      $(trac-mainnav-triplets $name)
EOT

}


trac-bannerpath-(){
    case $TRAC_INSTANCE in 
 dybsvn|mdybsvn) echo $ENV_HOME/logo/theta13_offline_software.png ;;
              *) echo $ENV_HOME/logo/trac_bannar_${TRAC_INSTANCE}.png ;;
    esac        
}

trac-bannerpath(){
   local path=$(trac-bannerpath-)
   [ -f $path ] && echo $path || echo -n
} 

trac-banner-triplet(){
   local msg="=== $FUNCNAME :"
   local default="header_logo:src:common/trac_banner.png"
   local path=$(trac-bannerpath)
   path=${path:-dummy}
   local name=$(basename $path)
   local htpath=$(trac-envpath)/htdocs/$name
   if [ -n "$TRAC_BANNER_TRIPLET_DEBUG" ]; then
      echo $msg this is screwed up if the permissions on the untarred dirs do not allow other access
      echo $msg path $path name $name
      ls -al $(dirname $path)
      echo $msg htpath $htpath
      ls -al $(dirname $htpath)
   fi
   [ ! -f "$path" ]   && echo $default && return 0   
   [ ! -f "$htpath" ] && echo $default && return 0 
   echo header_logo:src:site/$name
}

trac-placebanner(){
   local msg="=== $FUNCNAME :" 
   local path=${1:-$(trac-bannerpath)}
   [ ! -f "$path" ] && echo $msg no such path $path && return 1
   local dest=$(trac-envpath)/htdocs/$(basename $path)
   [ -f "$dest" ] && echo $msg already present at $dest && return 0

   local cmd="$SUDO cp -f $path $dest "
   echo $msg $cmd
   eval $cmd
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

trac-bitten-include(){
  local x=$(cat << EOX | tr "\n" "," 
lcgcmt/trunk
gaudi/trunk
relax/trunk
lhcb/trunk
dybgaudi/trunk
ldm/trunk
EOX)
 echo ${x:0:$((${#x}-1))}   ## just to remove the trailing comma
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

trac-bitten-xwords(){
 echo minor,trivial,minimal
}


trac-bitten-conf(){
  trac-configure $(cat << EOC
     bitten:dybinst.exclude_words:$(trac-bitten-xwords)
     bitten:dybinst.exclude_paths:
     bitten:dybinst.include_paths:$(trac-bitten-include)
     bitten:opt.dybinst.exclude_words:$(trac-bitten-xwords)
     bitten:opt.dybinst.exclude_paths:
     bitten:opt.dybinst.include_paths:$(trac-bitten-include)
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






