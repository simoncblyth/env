#
#      svn-x
#      svn-i
#      svn-s

#  ...  apache2 connection ....
#
#      svn-apache2-settings                      add the LoadModule lines to httpd.conf .. for mod_dav_svn mod_authz_svn and mod_python 
#
#      svn-apache2-conf                          write the location into SVN_APACHE2_CONF and plant the Include to it in httpd.conf
#        svn-apache2-location                    location "element" for httpd.conf to stdout
#        svn-apache2-authzaccess
#
#      svn-apache2-add-user name                 add a user to the htpasswd file , or change an existing users password
#      svn-apache2-open
#
#    svn-apache2-location-simpleauth-deprecated
#
#    ---------------
#
#     "svn status -u" from the working copy path of interest     
#           should show a single line :  "Status against revision:      8"
# 
#     test that the working copy is a clean revision with smth like:
#
#
# 
#
#    oops preferred version for trac is 1.2.3   ????  http://trac.edgewall.org/wiki/TracSubversion
#


#
# runtime settings including defining and adding SVN_HOME to the PATH are in scm_use.bash
#


svn-env(){


export PYTHON_PATH=$SVN_HOME/lib/svn-python:$PYTHON_PATH
export TRAC_EGG_CACHE=/tmp/trac-egg-cache

}



svn-x(){ scp $SCM_HOME/svn.bash ${1:-$TARGET_TAG}:$SCM_BASE; }

svn-i(){ . $SCM_HOME/svn.bash ; }

svn-info(){
  cat << EOI
  LOCAL_BASE $LOCAL_BASE
  SVN_NAME $SVN_NAME
  SVN_ABBREV $SVN_ABBREV
EOI
}


svn-apache2-settings(){
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







svn-apache2-info(){

    local msg="=== $FUNCNAME :"
    echo $msg APACHE2_BASE $APACHE2_BASE
	echo $msg APACHE2_LOCAL $APACHE2_LOCAL
	echo $msg APACHE2_BASE/SVN_APACHE2_AUTHZACCESS $APACHE2_BASE/$SVN_APACHE2_AUTHZACCESS



}





svn-apache2-conf(){

  ## raw SVN setup  

  local access=${1:-formlogin}
  if ([ "$access" == "formlogin" ] || [ "$access" == "httplogin" ]) then
     echo ======= svn-apache2-conf access $access ======= 
  else
    echo error access $access not handled
    return 1 
  fi
  
  $SUDO mkdir -p $APACHE2_BASE/$APACHE2_LOCAL

  svn-apache2-repos-location-write $APACHE2_BASE/$SVN_APACHE2_CONF $*
  apache2-conf-connect $SVN_APACHE2_CONF

  svn-apache2-authzaccess-write $APACHE2_BASE/$SVN_APACHE2_AUTHZACCESS dev

  svn-apache2-xslt-write $APACHE2_XSLT
  

  ## tracs 

  
  svn-apache2-tracs-location-write $APACHE2_BASE/$TRAC_APACHE2_CONF  $access 
  apache2-conf-connect $TRAC_APACHE2_CONF

  mkdir -p $TRAC_EGG_CACHE

  ## restart  

  if [ "$NODE_APPROACH" == "stock" ]; then
     echo === skip the check  
  else	 
     [ "$APACHE2_HOME/sbin" == $(dirname $(which apachectl)) ] || (  echo your PATH to apache2 executables is not setup correctly  && return ) 
  fi

  apache2-setport 6060   ##  $SCM_PORT  this is 80 (for client usage) but not appropiate at this level 

  apachectl configtest && echo restarting apache2 && $ASUDO apachectl restart || echo apachectl configtest failed

  #curl -o $APACHE2_HTDOCS/favicon.ico http://grid1.phys.ntu.edu.tw:6060/tracs/red/

}


svn-apache2-repos-location-write(){

  local conf=${1:-dummy-location}
  shift
  
  echo =============== writing svn-apache2-location output to $conf
  $ASUDO bash -lc "svn-apache2-repos-location $*  >  $conf"
  echo =============== cat $conf 
  cat $conf 

}







svn-apache2-tracs-location-write(){

  local conf=${1:-dummy-location}
  shift
  
  echo =============== writing svn-apache2-tracs-location output to $conf ... $*
  $ASUDO bash -lc "svn-apache2-tracs-location $* >  $conf"
  echo =============== cat $conf 
  cat $conf 


}


svn-apache2-authzaccess-update(){

  local msg="=== $FUNCNAME :"
  echo $msg this needs running after updating the users function 
  ## hmm ... if i split of the users defining  function into a file i could auto detect the need to do this update
  
  local authz=$APACHE2_BASE/$SVN_APACHE2_AUTHZACCESS
  [ ! -f $authz ] && echo $msg ABORT no authz $authz &&  return 1 
  
  ls -l $authz
  ASUDO=sudo svn-apache2-authzaccess-write $authz dev
  ls -l $authz
}



svn-apache2-authzaccess-write(){

  local authzaccess=${1:-dummy-authzaccess}
  shift
  
  echo =============== writing svn-apache2-authzaccess output to $authzaccess as root
  ## cannot use ASUDO="sudo -u $APACHE2_USER" directly as apache cannot access my .bash_profile
  $ASUDO bash -lc "svn-apache2-authzaccess $* >  $authzaccess"
  $ASUDO chown $APACHE2_USER $authzaccess
   ls -l $authzaccess
   echo =============== cat $authzaccess
  cat $authzaccess

}


svn-apache2-xslt-write(){

  local xslt=${1:-dummy-xslt}
  
  local iwd=$PWD
  
  if [ "$NODE_APPROACH" == "stock" ]; then
    echo === svn export the svn xslt for approach:  $NODE_APPROACH
	
	local resources_folder=$(dirname $xslt)
	$ASUDO mkdir -p $resources_folder
	
	cd $resources_folder
	$ASUDO rm -rf xslt
	$ASUDO svn export http://svn.collab.net/repos/svn/trunk/tools/xslt/ 
	
  else
  
     echo ============== placing stylesheets for raw SVN presentation into $xslt
     $ASUDO mkdir -p $xslt 
     $ASUDO cp -f $SVN_BUILD/tools/xslt/svnindex.* $xslt/ 

  fi
  ## correct a braindead absolute path 
  $ASUDO perl -pi -e 's|/svnindex.css|/resources/xslt/svnindex.css|' $xslt/svnindex.xsl 


  cd $iwd

}





svn-apache2-tracs-location(){

  local access=${1:-formlogin}
  if [ "$access" == "httplogin" ]; then
     c="" 
  elif [ "$access" == "formlogin" ]; then 
     c="#"
  else
     c="#ERROR access $access not handled "
  fi      

 if [ "$NODE_APPROACH" == "stock" ]; then
    local confprefix="/private/"
	c=""   ## uncomment for stock
  else
    local confprefix=""
  fi 




  local eggcache=$TRAC_EGG_CACHE
  $ASUDO chown $APACHE2_USER $eggcache 

cat << EOC

<Location /mpinfo>
   SetHandler mod_python
   PythonHandler mod_python.testhandler
   AllowOverride None
   Order Deny,Allow
   #  all local requests come in on 10.0.1.1 so cannot distinguish
   #Allow from 10.0.1.103
   #Allow from 10.0.1.104
   Allow from 10.0.1.1
   Deny from all
</Location>

<Directory $APACHE2_HTDOCS/test>
   AddHandler mod_python .py
   PythonHandler myscript
   PythonDebug On
</Directory>


<Location /tracs>
   SetHandler mod_python
   PythonPath "sys.path + ['$PYTHON_SITE']"
   PythonHandler trac.web.modpython_frontend 
   PythonOption TracEnvParentDir $SCM_FOLD/tracs
   PythonOption TracUriRoot /tracs
   PythonDebug On
   
   ## observe stylesheets inaccessible with msg about the 
   SetEnv PYTHON_EGG_CACHE $eggcache
   
   ## recent addition, reading between lines from http://trac.edgewall.org/wiki/TracMultipleProjectsSVNAccess
   # ... hmmm ... this is not the correct place ... should be in conf/trac.ini , or perhaps in global equivalent 
   #  
   #	 
   SVNParentPath $SVN_PARENT_PATH
   AuthzSVNAccessFile $SVN_APACHE2_AUTHZACCESS
   
</Location>

# when using AccountManagerPlugin this needs to be removed 
$c<LocationMatch "/tracs/[^/]+/login">
$c    AuthType Basic
$c    AuthName "svn-tracs"
$c    AuthUserFile $confprefix$SVN_APACHE2_AUTH
$c    Require valid-user
$c</LocationMatch>
$c
#  before AccounManagerPlugin is setup removing this causes ... 
# 500 Internal Server Error (Authentication information not available. 
#  Please refer to the <a href="/tracs/hottest/wiki/TracInstall#ConfiguringAuthentication" title="Configuring Authentication">installation documentation</a>.)
#


EOC
}




svn-apache2-repos-location(){


  local securitylevel=${1:-anon-or-real}

  if [ "$NODE_APPROACH" == "stock" ]; then
    local confprefix="/private/"
  else
    local confprefix=""
  fi 
   

cat << EOH
#
#      do not edit, created by svn.bash::svn-apache2-location 
#       http://svnbook.red-bean.com/en/1.0/ch06s04.html
#
#   NB no need to restart apache2 on creating new repositories
#
#
<Location /repos>
      DAV svn
      SVNParentPath $SVN_PARENT_PATH
      ## allow raw apache+svn to provide a list of repositories
      SVNListParentPath on
      SVNIndexXSLT /resources/xslt/svnindex.xsl

      # access policy file
      AuthzSVNAccessFile $confprefix$SVN_APACHE2_AUTHZACCESS

      
   
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
      AuthUserFile $confprefix$SVN_APACHE2_AUTH
                  
</Location>
EOT

}


svn-co-test(){

   local n=${1:-newtest}
   local u=${2:-admin}
   local p=${3:-wrong}
   
   if ([ "$n" == "newtest" ] || [ "$n" == "hottest" ]) then
      local cmd="cd /tmp ; rm -rf $n ; svn --username $u --password $p  co  http://localhost/repos/$n/ "   
       echo $cmd
       eval $cmd
   else
     echo name $n not accepted
   fi
}


svn-apache2-authzaccess(){

  local finelevel=${1:-dev}
 
  # name of the environment repository 
  local envbase=$ENV_BASE  

cat << EOA
#
#      do not edit, created by svn.bash::svn-apache2-authzaccess, svn-apache2-conf
#      http://svnbook.red-bean.com/en/1.0/ch06s04.html
#
#  securitylevel $securitylevel
#   
[groups]
sync = ntusync
member = simon, dayabay 
user = blyth, thho, chwang
admin = dayabaysoft, admin
heprezmember = simon, cjl, tosi, cecilia
heprezuser = blyth
aberdeen = blyth, thho, jimmy, antony, soap, bzhu, wei, adair, chwang

EOA

if [ "$finelevel" == "example" ]; then

cat << EOC
#  read access for everyone
[/]
* = r

[red:/trunk]
@member = r
@user = rw
@admin = rw

# caution this matches the path in every repository 
[/trunk]
@member = r
@user = rw
@admin = rw

EOC

elif [ "$finelevel" == "dev" ]; then

cat << EOC
# empty fine grained permissions gives no access to anyone !!!
# 
# read access to everyone and write access for admin and user groups
# for repos : $envbase newtest  
#
# no access to everyone and readwrite acees for admin and 
#

#[/]
#@admin = rw

[env:/]
* = r
@member = r
@user = rw 
@admin = rw
@aberdeen = rw

[workflow:/]
* = r
@member = r
@user = rw 
@admin = rw

[ApplicationSupport:/]
* = r
@member = r
@user = rw 
@admin = rw

[heprez:/]
* = r
@heprezmember = rw
@heprezuser = rw 
@admin = rw

[tracdev:/]
* = r
blyth = rw
@user = r 
@admin = rw


# Nb this is declaring anyone can read the content of this repository ...
# so must allow anonymous to get thru at the upper level ...
[newtest:/]
* = r
@member = r
@user = rw 
@admin = rw


# force authenticated 


[dyw:/]
@member = r
blyth = rw
@user = rw 
@admin = rw

[dybsvn:/]
@sync = rw
@member = r 
@user = r 
@admin = r

[aberdeen:/]
@member = r
@aberdeen = rw 
@admin = rw


 
EOC

else

cat << EOC
# WARNING securitylevel not handled
EOC

fi




}










svn-apache2-location-simpleauth-deprecated(){

## NB no need to restart apache2 on creating new repositories
cat << EOC
<Location /repos>
      DAV svn
      SVNParentPath $SVN_PARENT_PATH
      SVNIndexXSLT /resources/xslt/svnindex.xsl

      AuthType Basic
      AuthName "svn-repos"
      AuthUserFile $SVN_APACHE2_AUTH

# For any operations other than these, require an authenticated user.
      <LimitExcept GET PROPFIND OPTIONS REPORT>
	     Require valid-user
      </LimitExcept>

</Location>
EOC

}



svn-global-ignores(){

cat << EOI
#  uncomment global-ignores in [miscellany] section of
#     $HOME/.subversion/config
#  setting it to : 
#
global-ignores = setup.sh setup.csh cleanup.sh cleanup.csh Linux-i686* Darwin* InstallArea Makefile load.C
#
#    NB there is no whitespace before "global-ignores"
# 
#  after this   
#        svn status -u 
#  should give a short enough report to be useful
#
EOI

echo vi $HOME/.subversion/config


}








