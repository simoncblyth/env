
svnsetup-src(){ echo svn/svnconf/svnsetup.bash ; }
svnsetup-source(){ echo ${BASH_SOURCE:-$(env-home)/$(svnsetup-src)} ; }
svnsetup-vi(){    vi $(svnsetup-source) ; }
authz-(){ . $(env-home)/svn/svnconf/authz.bash && authz-env $* ; }

svnsetup-usage(){

  cat << EOU 

   $BASH_SOURCE

     For infrequently used setup of svn + apache ...  

     svnsetup-apache <path/to/apache/conf/folder>       defaults to: $(svn-setupdir)  
                invokes the below funcs
                to create the apache conf files and hooks them up to httpd.conf
                by appending an Include 

     test with
        SUDO=sudo svnsetup-apache /tmp/env/svnsetup/apache
        
     use with ... 
         SUDO=sudo svnsetup-apache 

     Or for just authz updates ... the usual change on adding new users :
     
        1) modify svn/svnconf/authz.bash giving appropriate permissions to the new
           usernames
        2) update the authz file
              svn-
              svnsetup-
              SUDO=sudo svnsetup-authz-update
        3) add corresponding users thru the trac AccountManager  








     svnsetup-tracs <path/to/tracs.conf>
     svnsetup-repos <path/to/repos.conf> 
     svnsetup-svn   <path/to/svn.conf>    ## for IHEP svn topfold of /svn
     svnsetup-authz <path/to/authz.conf>
     
           writes to /tmp/env/svnsetup/{tracs,repos,authz}.conf 
           if a path is given then copies the temporary to it using 
           SUDO:$SUDO
           
           
           authz could potentially be managed by a trac plugin
           
            (extremely simple approach, just editing file thru web interface
              ... so simple almost not worth doing, just edit the file with 
              the function is more traceable
            )
               http://www.trac-hacks.org/wiki/TracSvnAuthzPlugin
            (involved approach, must be parsing the file
              .... to get it working on 0.11 would have to diddle with multiple
              patches
              )    
               http://www.trac-hacks.org/wiki/SvnAuthzAdminPlugin            
           
           
           
     svnsetup-tracs-
     svnsetup-repos-
     svnsetup-authz-
          cats to stdout filling in the blanks from the environment 
     
     svnsetup-location-
            used internally by the above funcs





     svnsetup-put-users-to-node target-node 
     svnsetup-get-users-from-h 
    
    



     svnsetup-modules 
           add the requisite modules for apache+svn runninf to httpd.conf 
           THIS IS NOW HANDLED IN APACHEBUILD ???     
         
     svnsetup-xslt
           get the xsl/css for prettier raw svn               


    TODO : 
       ownership...
           





EOU


}


svnsetup-env(){
  elocal-
  python-
  apache-
}

svnsetup-dir(){
  echo $(apache-confdir)/svnsetup-deprecated-use-svn-setupdir 
}

svnsetup-tmp(){
  echo /tmp/env/${FUNCNAME/-*/}/apache
}



svnsetup-authz-update(){

  local msg="=== $FUNCNAME :"
  local authz=$(svn-authzpath)
  echo $msg updating $authz
  svnsetup-authz $authz

}


svnsetup-selinux-persist-(){
	cat << EOC

	sudo /sbin/restorecon -F -r -n -vv $(local-scm-fold)      ## dry run
	sudo /sbin/restorecon -F -r    -vv $(local-scm-fold)      ## standardize labels

        sudo /usr/sbin/semanage fcontext -a -t httpd_sys_content_t "$(local-scm-fold)/repos(/.*)?"       ## change the standard labels
        sudo /usr/sbin/semanage fcontext -a -t httpd_sys_content_t "$(local-scm-fold)/tracs(/.*)?" 
        sudo /usr/sbin/semanage fcontext -a -t httpd_sys_content_t "$(local-scm-fold)/svn(/.*)?" 
        sudo /usr/sbin/semanage fcontext -a -t httpd_sys_content_t "$(local-scm-fold)/conf(/.*)?" 

        sudo /sbin/restorecon -F -r -n -vv $(local-scm-fold)      ## dry run ... see what labels are going to change 
	sudo /sbin/restorecon -F -r    -vv $(local-scm-fold)      ## apply the change in standardization 


EOC
}


svnsetup-selinux-(){
  cat << EOC
sudo chcon -R -t httpd_sys_content_t $(local-scm-fold)/repos
sudo chcon -R -t httpd_sys_content_t $(local-scm-fold)/tracs
sudo chcon -R -t httpd_sys_content_t $(local-scm-fold)/svn
sudo chcon -R -t httpd_sys_content_t $(local-scm-fold)/conf
EOC

 python-
 [ "$(python-mode)" == "source" ] && echo sudo chcon -R -t httpd_sys_content_t $(python-home)/bin

 apache-
 [ "$(apache-mode)" == "source" ] && echo sudo chcon -R -t httpd_config_t $(apache-confdir)

}

svnsetup-selinux-persist(){
   local cmd
   svnsetup-selinux-persist- | while read cmd ; do
      echo $cmd
      eval $cmd
   done
}

svnsetup-selinux(){
   local cmd
   svnsetup-selinux- | while read cmd ; do 
      echo $cmd
      eval $cmd
   done
}

svnsetup-sysapache(){

   local msg="=== $FUNCTION : "
   apache-
   [ "$(apache-mode)" != "system" ] && echo $msg ABORT this is for system apache only ... perhaps you should use svnsetup-apache && return 1

   local base=$(apache-confd)
   [ ! -d "$base" ] && echo $msg ABORT apache-confd $base does not exist && return 2    

   svnsetup-tracs $base/tracs.conf 
   svnsetup-repos $base/repos.conf anon-or-real repos
   svnsetup-svn   $base/svn.conf   anon-or-real svn      ## for recovered IHEP repositories

   local authz=$(svn-authzpath)
   svnsetup-authz $(dirname $authz)/$(basename $authz)
  
}

svnsetup-apache(){


   local msg="=== $FUNCNAME :"
   apache-   
   [ "$(apache-mode)" != "source" ] && echo $msg ABORT this is for source apache only ... perhaps you should use svnsetup-sysapache && return 1

   local def=$(svn-setupdir)
   local base=${1:-$def}
   

   if [ "$base" == "$def" ]; then
      echo $msg setting ownership of $base
      $SUDO mkdir -p $base 
      svnsetup-chown $base 
      
      echo addline... Include $base/setup.conf
      apache-
      apache-addline "Include $base/setup.conf"
   
   else
      mkdir -p $base
   fi

   svnsetup-tracs $base/tracs.conf 
   svnsetup-repos $base/repos.conf anon-or-real repos
   svnsetup-svn   $base/svn.conf   anon-or-real svn      ## for recovered IHEP repositories
   svnsetup-setup $base/setup.conf
   
   local authz=$(svn-authzpath)
   svnsetup-authz $base/$(basename $authz)
   
  
   ## do the common config used for all instances
   trac-
   trac-inherit-setup  

}





svnsetup-chown-deprecated(){

   local path=$1
   local msg="=== $FUNCNAME :"
   [ "$SUDO" == "" ] && echo $msg not a sudoer skipping && return 1
   
   shift
   local user=$(apache-user)
   local group=$(apache-group)
   
   case $NODE_TAG in 
     G) $SUDO chown $* $user:$group $path ;;
     *) $SUDO chown $* $user:$group $path ;;
   esac
    
}



svnsetup-tracs(){ svnsetup-location- $FUNCNAME $* ; }
svnsetup-repos(){ svnsetup-location- $FUNCNAME $* ; }
svnsetup-svn(){   svnsetup-location- $FUNCNAME $* ; }
svnsetup-authz(){ svnsetup-location- $FUNCNAME $* ; }
svnsetup-setup(){ svnsetup-location- $FUNCNAME $* ; }


svnsetup-location-(){

  local msg="=== $FUNCNAME :"
  local flavor=${1/*-/}
  shift
  case $flavor in 
    tracs) echo -n ;;
repos|svn) echo -n ;;
    authz) echo -n ;;
    setup) echo -n ;;
        *) echo $msg ABORT this should be invoked by svnsetup-tracs/repos .. && return 1
  esac
  
  local tmp=/tmp/env/${FUNCNAME/-*/}/$flavor.conf && mkdir -p $(dirname $tmp)
  
  local path=$1
  shift  
  
  svnsetup-$flavor- $* >  $tmp 
  echo $msg $flavor wrote tmp $tmp 
  ls -l $tmp
  #cat $tmp
  
  [ -z $path ] && return 0
  


  echo $msg $flavor copying tmp $tmp to $path with SUDO [$SUDO]
  $SUDO cp $tmp $path
  
  #local user=$(apache-user)   
  echo $msg $flavor setting ownership to $user   
  apache-chown $path
     
  ls -l $path
  #cat $path 
     
}


svnsetup-setup-(){

  local msg="=== $FUNCNAME : "
  local dir=$(svn-setupdir)

cat << EOU 
#
#  $msg $BASH_SOURCE $(date)
#
#   \$(svn-setupdir)     :  $(svn-setupdir)
#
Include $dir/repos.conf
Include $dir/svn.conf
Include $dir/tracs.conf

EOU


}





svnsetup-tracs-(){

 local msg="=== $FUNCNAME : " 
 
 local c="#"
 local b=""
  
cat << EOC
#
#    $msg $BASH_SOURCE  $(date)
#
#     c:[$c] b:[$b] 
#   
#   \$(apache-htdocs)         :  $(apache-htdocs) 
#   \$(python-site)           :  $(python-site)
#   \$(svn-userspath)         :  $(svn-userspath)
#   \$SCM_FOLD                :  $SCM_FOLD
#
#

<Location /mpinfo>
   SetHandler mod_python
   PythonHandler mod_python.testhandler
   Order Deny,Allow
   #  all local requests come in on 10.0.1.1 so cannot distinguish
   #Allow from 10.0.1.103
   #Allow from 10.0.1.104
   Allow from 10.0.1.1
   Allow from 140.112.102.77
   Deny from all
</Location>

<Directory $(apache-htdocs)/test>
   AddHandler mod_python .py
   PythonHandler myscript
   PythonDebug On
</Directory>


<Location /tracs>
   SetHandler mod_python
   PythonPath "sys.path + ['$(python-site)']"
   PythonHandler trac.web.modpython_frontend 
   PythonOption TracEnvParentDir $SCM_FOLD/tracs
   PythonOption TracUriRoot /tracs
   PythonDebug On
   
   ## observe stylesheets inaccessible with msg about the 
   SetEnv PYTHON_EGG_CACHE /tmp/trac-egg-cache
   
   ## recent addition, reading between lines from http://trac.edgewall.org/wiki/TracMultipleProjectsSVNAccess
   # ... hmmm ... this is not the correct place ... should be in conf/trac.ini , or perhaps in global equivalent 
   #  
   #	 
   ## SVNParentPath \$SVN_PARENT_PATH
   ## AuthzSVNAccessFile \$SVN_APACHE2_AUTHZACCESS
   
</Location>


# when not using bitten this should be removed 
$b<LocationMatch "/tracs/[^/]+/builds">
$b    AuthType Basic
$b    AuthName "svn-tracs"
$b    AuthUserFile $(svn-userspath)
$b    Require valid-user
$b</LocationMatch>
$b

<Location /logs>
    AuthType Basic
    AuthName "svn-tracs"
    AuthUserFile  $(svn-userspath)
    Require valid-user
</Location>


# when using AccountManagerPlugin this needs to be removed 
$c<LocationMatch "/tracs/[^/]+/login">
$c    AuthType Basic
$c    AuthName "svn-tracs"
$c    AuthUserFile $(svn-userspath)
$c    Require valid-user
$c</LocationMatch>
$c
#  before AccounManagerPlugin is setup removing this causes ... 
# 500 Internal Server Error (Authentication information not available. 
#  Please refer to the <a href="/tracs/hottest/wiki/TracInstall#ConfiguringAuthentication" title="Configuring Authentication">installation documentation</a>.)
#

EOC

}

svnsetup-svn-(){
  svnsetup-repos- $*  
}

svnsetup-repos-(){

  local securitylevel=${1:-anon-or-real}
  local topfold=${2:-repos} 
   
cat << EOH


#    $msg $BASH_SOURCE  $(date)
#
#     securitylevel:[$securitylevel] 
#     topfold : [$topfold]
#
#     \$SCM_FOLD                 : $SCM_FOLD
#     \$(svn-authzpath)          : $(svn-authzpath)
#     \$(svn-userspath)          : $(svn-userspath)
#
#    http://svnbook.red-bean.com/en/1.0/ch06s04.html
#     NB no need to restart apache2 on creating new repositories
#
#
<Location /$topfold>
      DAV svn
      SVNParentPath $SCM_FOLD/$topfold
      ## allow raw apache+svn to provide a list of repositories
      SVNListParentPath on
      ##SVNIndexXSLT /resources/xslt/svnindex.xsl

      # access policy file
      AuthzSVNAccessFile $(svn-authzpath)

      # securitylevel $securitylevel
EOH
   
   
  if [ "$securitylevel" == "anon-or-real" ]; then
   
      cat << EOS   
      #  try anonymous access first, resort to real authentication if necessary.
      Satisfy Any
      Require valid-user
EOS
   
  elif [ "$securitylevel" == "authenticated-only" ]; then
      cat << EOS
      # only authenticated users may access the repository
      Require valid-user 
EOS

  else
      cat << EOS
      ## WARNING securitylevel not handled
EOS
  fi
   
  cat << EOT   
  
      # how to authenticate a user
      AuthType Basic
      AuthName "svn-repos"
      # users file
      AuthUserFile $(svn-userspath)
                  
</Location>
EOT

}



svnsetup-authz-(){

  local authz_=$ENV_HOME/svn/svnconf/authz.bash
  [ ! -f $authz_ ] && echo $msg ABORT no authz $authz_ && sleep 10000000000000
  . $authz_
  authz

}












svnsetup-modules-deprecated(){
  echo ======= the below LoadModule directives are needed for svn 
  cat << EOS
LoadModule dav_svn_module     $APACHE2_SO/mod_dav_svn.so
LoadModule authz_svn_module   $APACHE2_SO/mod_authz_svn.so
#LoadModule wsgi_module        $APACHE2_SO/mod_wsgi.so
LoadModule python_module   $APACHE2_SO/mod_python.so

EOS

  apache2-add-module dav_svn 
  apache2-add-module authz_svn 
 #apache2-add-module wsgi
  apache2-add-module python

  echo $SUDO vi $APACHE2_CONF 
}




svnsetup-xslt(){

  local msg="=== $FUNCNAME :"
  local tmp=/tmp/env/${FUNCNAME/-*/} && mkdir -p $tmp
  local xslt=${1:-$tmp}
  local iwd=$PWD
  
  if [ "$NODE_APPROACH" == "stock" ]; then
  
    echo $msg svn export the svn xslt for NODE_APPROACH:$NODE_APPROACH
	local resources_folder=$(dirname $xslt)
	$SUDO mkdir -p $resources_folder
	cd $resources_folder
	$SUDO rm -rf xslt
	$SUDO svn export http://svn.collab.net/repos/svn/trunk/tools/xslt/ 
	
  else
  
     echo $msg  placing stylesheets for raw SVN presentation into $xslt
  
     svnbuild-
     local build=$(svnbuild-dir)/tools/xslt
     [ ! -d $build ] && echo $msg ABORT no build $build && return 1
           
     $SUDO mkdir -p $xslt 
     $SUDO cp -f $build/svnindex.* $xslt/ 
     
  fi
  
  ## correct a braindead absolute path 
  
  $SUDO perl -pi -e 's|/svnindex.css|/resources/xslt/svnindex.css|' $xslt/svnindex.xsl 

  cd $iwd
}



svnsetup-put-users-to-node(){

  local t=${1:-C}
  local msg="=== $FUNCNAME :"
  [ "$NODE_TAG" != "H" ] && echo $msg ABORT only applicable on H not  NODE_TAG $NODE_TAG && return 1 
  
  local user=$(NODE_TAG=$t apache-user)
  local conf=$(NODE_TAG=$t svn-userspath)  
  local tmp=/tmp/env/svnsetup/$(basename $conf)
  
  local cmd=$(cat << EOC 
      ssh $t "mkdir -p $(dirname $tmp)" ;  
      scp $(svn-userspath) $t:$tmp ; 
      ssh $t "sudo cp $tmp $conf ; sudo chown $user:$user $conf  " 
EOC)
        
  echo $cmd
  eval $cmd 

}


svnsetup-get-users-from-h(){
   svnsetup-get-users-from-node H
}

svnsetup-get-users-from-node(){

   local tag=${1:-H}
   
   local msg="=== $FUNCNAME :"
   [ "$NODE_TAG" == "H" ] && echo $msg ABORT not applicable on NODE_TAG $NODE_TAG && return 1 
   
   
   local cmd="scp $tag:$(NODE_TAG=$tag svn-userspath) $(svn-userspath)"
   echo $msg $cmd
   eval $cmd

   svnsetup-chown $users
}






