
svnsetup-usage(){

  cat << EOU 

   $BASH_SOURCE

     For infrequently used setup of svn + apache ...  

     svnsetup-apache <path/to/apache/conf/folder>       defaults to: $(svnsetup-tmp)  
                invokes the below funcs
                to create the apache conf files

     test with
        ASUDO=sudo svnsetup-apache /tmp/env/svnsetup/apache
        
     use with ... 
         ASUDO=sudo svnsetup-apache 

     svnsetup-tracs <path/to/tracs.conf>
     svnsetup-repos <path/to/repos.conf> 
     svnsetup-authz <path/to/authz.conf>
     
           writes to /tmp/env/svnsetup/{tracs,repos,authz}.conf 
           if a path is given then copies the temporary to it using 
           ASUDO:$ASUDO
           
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
  echo $(apache-confdir)/svnsetup 
}

svnsetup-tmp(){
  echo /tmp/env/${FUNCNAME/-*/}/apache
}


svnsetup-put-users-to-node(){

  local t=${1:-C}
  local msg="=== $FUNCNAME :"
  [ "$NODE_TAG" != "H" ] && echo $msg ABORT only applicable on H not  NODE_TAG $NODE_TAG && return 1 
  
  local tmp=/tmp/env/svnsetup
  local user=$(NODE_TAG=$t apache-user)
  local conf=$(NODE_TAG=$t svnsetup-dir)/users.conf
  
  local cmd=$(cat << EOC 
      ssh $t "mkdir -p $tmp" ;  
      scp $(apache-confdir)/svn-apache2-auth $t:$tmp/users.conf ; 
      ssh $t "sudo cp $tmp/users.conf $conf ; sudo chown $user:$user $conf  " 
EOC)
        
  echo $cmd
  eval $cmd 

}


svnsetup-get-users-from-h(){

   local msg="=== $FUNCNAME :"
   [ "$NODE_TAG" == "H" ] && echo $msg ABORT not applicable on NODE_TAG $NODE_TAG && return 1 
   
   local users=$(svnsetup-dir)/users.conf
   local cmd="scp H:$(NODE_TAG=H apache-confdir)/svn-apache2-auth $users"
   echo $msg $cmd
   eval $cmd

   svnsetup-chown $users
}


svnsetup-apache(){

   local msg="=== $FUNCNAME :"
   local def=$(svnsetup-dir)
   local base=${1:-$def}
   

   if [ "$base" == "$def" ]; then
      echo $msg setting ownership of $base
      $ASUDO mkdir -p $base 
      svnsetup-chown $base 
      
      apache-
      apache-addline "Include $base/setup.conf"
      
   fi

   svnsetup-tracs $base/tracs.conf 
   svnsetup-repos $base/repos.conf
   svnsetup-setup $base/setup.conf
   
   svnsetup-authz $base/authz.conf
   
   
   
}





svnsetup-chown(){
   local path=$1
   shift
   local user=$(apache-user)
   
   case $NODE_TAG in 
     G) $ASUDO chown $* $user:$user $path ;;
     *) $ASUDO chown $* $user:$user $path ;;
   esac
    
}



svnsetup-tracs(){ svnsetup-location- $FUNCNAME $* ; }
svnsetup-repos(){ svnsetup-location- $FUNCNAME $* ; }
svnsetup-authz(){ svnsetup-location- $FUNCNAME $* ; }
svnsetup-setup(){ svnsetup-location- $FUNCNAME $* ; }



svnsetup-location-(){

  local msg="=== $FUNCNAME :"
  local flavor=${1/*-/}
  shift
  case $flavor in 
    tracs) echo -n ;;
    repos) echo -n ;;
    authz) echo -n ;;
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
  


  echo $msg $flavor copying tmp $tmp to $path with ASUDO [$ASUDO]
  $ASUDO cp $tmp $path
  
  local user=$(apache-user)   
  echo $msg $flavor setting ownership to $user   
  svnsetup-chown $path
     
  ls -l $path
  #cat $path 
     
}


svnsetup-setup-(){

  local msg="=== $FUNCNAME : "
  local dir=$(svnsetup-dir)

cat << EOU 
#
#  $msg $BASH_SOURCE $(date)
#
#   \$(svnsetup-dir)     :  $(svnsetup-dir)
#
Include $dir/repos.conf
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
#   \$(svnsetup-dir)          :  $(svnsetup-dir)
#   \$SCM_FOLD                :  $SCM_FOLD
#
#

<Location /mpinfo>
   SetHandler mod_python
   PythonHandler mod_python.testhandler
   AllowOverride None
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
$b    AuthUserFile $(svnsetup-dir)/users.conf
$b    Require valid-user
$b</LocationMatch>
$b


# when using AccountManagerPlugin this needs to be removed 
$c<LocationMatch "/tracs/[^/]+/login">
$c    AuthType Basic
$c    AuthName "svn-tracs"
$c    AuthUserFile $(svnsetup-dir)/users.conf
$c    Require valid-user
$c</LocationMatch>
$c
#  before AccounManagerPlugin is setup removing this causes ... 
# 500 Internal Server Error (Authentication information not available. 
#  Please refer to the <a href="/tracs/hottest/wiki/TracInstall#ConfiguringAuthentication" title="Configuring Authentication">installation documentation</a>.)
#

EOC

}




svnsetup-repos-(){

  local securitylevel=${1:-anon-or-real}
   
cat << EOH

#
#    $msg $BASH_SOURCE  $(date)
#
#     securitylevel:[$securitylevel] 
#
#     \$SCM_FOLD                 : $SCM_FOLD
#     \$(svnsetup-dir)           : $(svnsetup-dir)
#
#    http://svnbook.red-bean.com/en/1.0/ch06s04.html
#     NB no need to restart apache2 on creating new repositories
#
#
<Location /repos>
      DAV svn
      SVNParentPath $SCM_FOLD/repos
      ## allow raw apache+svn to provide a list of repositories
      SVNListParentPath on
      SVNIndexXSLT /resources/xslt/svnindex.xsl

      # access policy file
      AuthzSVNAccessFile $(svnsetup-dir)/authz.conf

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
      AuthUserFile $(svnsetup-dir)/users.conf
                  
</Location>
EOT

}



svnsetup-authz-(){

  local authz_=$(dirname $BASH_SOURCE)/authz.bash
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
	$ASUDO mkdir -p $resources_folder
	cd $resources_folder
	$ASUDO rm -rf xslt
	$ASUDO svn export http://svn.collab.net/repos/svn/trunk/tools/xslt/ 
	
  else
  
     echo $msg  placing stylesheets for raw SVN presentation into $xslt
  
     svnbuild-
     local build=$(svnbuild-dir)/tools/xslt
     [ ! -d $build ] && echo $msg ABORT no build $build && return 1
           
     $ASUDO mkdir -p $xslt 
     $ASUDO cp -f $build/svnindex.* $xslt/ 
     
  fi
  
  ## correct a braindead absolute path 
  
  $ASUDO perl -pi -e 's|/svnindex.css|/resources/xslt/svnindex.css|' $xslt/svnindex.xsl 

  cd $iwd
}






