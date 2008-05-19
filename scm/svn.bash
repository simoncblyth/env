
svn-env(){
#
# runtime settings including defining and adding SVN_HOME to the PATH are in scm_use.bash
#

 
  elocal-
  local SVN_NAME=subversion-1.4.0
  local SVN_ABBREV=svn
  
  if [ "$NODE_APPROACH" != "stock" ]; then
	  export SVN_HOME=$SYSTEM_BASE/$SVN_ABBREV/$SVN_NAME
	  export PYTHON_PATH=$SVN_HOME/lib/svn-python:$PYTHON_PATH
      svn-path
  fi	  

  export SVN_PARENT_PATH=$SCM_FOLD/repos

  ## just needed by svn.bash
  SVN_APACHE2_CONF=$APACHE2_LOCAL/svn.conf

  ## these are needed by both SVN + Trac  
  SVN_APACHE2_AUTH=$APACHE2_LOCAL/svn-apache2-auth
  SVN_APACHE2_AUTHZACCESS=$APACHE2_LOCAL/svn-apache2-authzaccess
   
   export TRAC_EGG_CACHE=/tmp/trac-egg-cache
   
}


svn-path(){

  local msg="=== $FUNCNAME :"
  if [ -z $SVN_HOME ]; then 
     echo $msg skipping as no SVN_HOME
  else
	 export DYLD_LIBRARY_PATH=$SVN_HOME/lib/svn-python/svn:$DYLD_LIBRARY_PATH
     export DYLD_LIBRARY_PATH=$SVN_HOME/lib/svn-python/libsvn:$DYLD_LIBRARY_PATH
     export PATH=$SVN_HOME/bin:$PATH
  fi

}





svn-usage(){
  cat << EOU
  
svn-apache2-conf    
                         pull together the below funcs with coordinated paths


ASUDO= svn-apache2-conf-   
                          generate the conf fragments into /tmp to verify before doing above



svn-apache2-settings                            
                          top level svn thru apache requirements 
						  add the LoadModule lines to httpd.conf .. for mod_dav_svn mod_authz_svn and mod_python 

svn-apache2-repos-location-write <path> <param>     
                          write the repos location by invoking the below

svn-apache2-repos-location  <anon-or-real|authenticated-only>    
					      emit the block  

svn-apache2-xslt-write <path>                  
                          get and place the xsl needed by raw svn viewing


svn-apache2-tracs-location-write <path>        
                          write the tracs location by invoking the below 

svn-apache2-tracs-location <httplogin|formlogin>               

svn-apache2-authzaccess-write  <path> <mode>   

svn-apache2-authzaccess
                          emit the fine grained permissions file with users and groups defintions

 
svn-apache2-authzaccess-update        
                           needs running after updating users function  
   


# svn-apache2-add-user name                
#                        add a user to the htpasswd file , or change an existing users password

	
	  
EOU

}




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



svn-apache2-conf-(){

	local msg="=== $FUNCNAME:"
    local access=${1:-formlogin}
	local base=${2:-/tmp/$FUNCNAME}
	
	shift 
	shift
	
	case $access in
	  formlogin)  echo $msg $access $base ;; 
	  httplogin)  echo $msg $access $base ;;
	          *)  echo $msg access $access not supported && return 1 ;;
	esac
	
	## raw SVN setup
	
	$ASUDO mkdir -p $base/$APACHE2_LOCAL
	
	svn-apache2-repos-location-write $base/$SVN_APACHE2_CONF $*
	svn-apache2-authzaccess-write    $base/$SVN_APACHE2_AUTHZACCESS dev
	
	## tracs 
    svn-apache2-tracs-location-write $base/$TRAC_APACHE2_CONF  $access 


}



svn-apache2-conf(){

  local access=${1:-formlogin}
  
  svn-apache2-conf- $access $APACHE2_BASE 

 ## the files are already Included curtesy of heprez-/apache-/apache-conf-heprez
 ## apache2-conf-connect $SVN_APACHE2_CONF
 ## apache2-conf-connect $TRAC_APACHE2_CONF
 
  svn-apache2-xslt-write $APACHE2_XSLT
  
  mkdir -p $TRAC_EGG_CACHE
  $ASUDO chown $APACHE2_USER $eggcache 

  ## restart  

  if [ "$NODE_APPROACH" == "stock" ]; then
     echo === skip the check  
  else	 
     [ "$APACHE2_HOME/sbin" == $(dirname $(which apachectl)) ] || (  echo your PATH to apache2 executables is not setup correctly  && return ) 
  fi

  ##apache2-setport 6060   ##  $SCM_PORT  this is 80 (for client usage) but not appropiate at this level 
  ##apachectl configtest && echo restarting apache2 && $ASUDO apachectl restart || echo apachectl configtest failed
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

  local msg="# === $FUNCNAME : " 
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
	#c=""   ## uncomment for stock
  else
    local confprefix=""
  fi 

  echo $msg access:[$access] c:[$c] NODE_APPROACH:[$NODE_APPROACH]


 
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
   SetEnv PYTHON_EGG_CACHE $TRAC_EGG_CACHE
   
   ## recent addition, reading between lines from http://trac.edgewall.org/wiki/TracMultipleProjectsSVNAccess
   # ... hmmm ... this is not the correct place ... should be in conf/trac.ini , or perhaps in global equivalent 
   #  
   #	 
   ## SVNParentPath $SVN_PARENT_PATH
   ## AuthzSVNAccessFile $SVN_APACHE2_AUTHZACCESS
   
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

  local tw="thho, bhzu, wei, adiar, chwang"
  local hk="jimmy, antony, soap"

cat << EOA
#
#      do not edit, created by svn.bash::svn-apache2-authzaccess, svn-apache2-conf
#      http://svnbook.red-bean.com/en/1.0/ch06s04.html
#
#  securitylevel $securitylevel
#   
[groups]

sync = ntusync
dyuser = blyth, $tw, $hk, dayabay

evuser = simon, dayabay
evdev = blyth, $tw, $hk 
evadmin = blyth, dayabaysoft, admin 

abuser = simon, dayabay
abdev = blyth, $tw, $hk
abadmin = blyth

hzuser = simon, cjl, tosi, cecilia
hzdev = blyth
hzadmin = blyth

tduser = simon
tddev = blyth
tdadmin = blyth

wfuser = simon
wfdev = blyth
wfadmin = blyth


# force authenticated 
[dybsvn:/]
@sync = rw
@dyuser = r 

[env:/]
* = r
@evuser = r
@evdev = rw 
@evadmin = rw

[aberdeen:/]
@abuser = r
@abdev = rw 
@abadmin = rw

[heprez:/]
* = r
@hzuser = rw
@hzdev = rw 
@hzadmin = rw

[tracdev:/]
* = r
@tduser = r
@tddev = rw 
@tdadmin = rw

[workflow:/]
@wfuser = r
@wfdev = rw 
@wfadmin = rw

[ApplicationSupport:/]
@wfuser = r
@wfdev = rw 
@wfadmin = rw
 
EOA


}



svn-apache2-authzaccess-example(){




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


# Nb this is declaring anyone can read the content of this repository ...
# so must allow anonymous to get thru at the upper level ...
[newtest:/]
* = r
@member = r
@user = rw 
@admin = rw




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








