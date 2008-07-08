
svn-apache2-usage(){
  cat << EOU
 
    CAUTION WHILE DOING THIS ... IF YOU GET THE CONFIG WRONG SUCH THAT IT MAKES YOU LOOSE PERMISSION
	TO THE REPOSITORY THEN YOU WILL NOT BE ABLE TO COMMIT CHANGES ...SEE   svn-tmp-cp IF IN A PINCH 
  
  
ASUDO=sudo svn-apache2-conf    
                         pull together the below funcs with coordinated paths


ASUDO= svn-apache2-conf-   
                          generate the conf fragments into /tmp/env/ to verify before doing above, also allows them
						  to save the day on subsequent screwup with svn-tmp-cp


svn-apache2-settings                            
                          top level svn thru apache requirements 
						  add the LoadModule lines to httpd.conf ..  mod_dav_svn mod_authz_svn and mod_python 

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

	
	
 Typical Usage .. when setting SVN permissions of users  :

   env-
   svn-apache2-
   svn-apache2-usage

 Test the setup ... writing to /tmp/env
          ASUDO= svn-apache2-conf-   

  If they look OK then ... 
          ASUDO=sudo svn-apache2-conf
          ls -l $APACHE__LOCAL/	
	
      NOTE THIS DOES NOT ADD THE USERS 
       ... DO THAT THRU THE webadmin INTERFACE MANAGING THE  svn-apache2-auth FILE	
	
	  
EOU

}



svn-apache2-env(){
#
# runtime settings including defining and adding SVN_HOME to the PATH are in scm_use.bash
#

 
  elocal-
  #apache2-
  apache-
  trac-

  svn-apache2-base 

}


svn-apache2-base(){

  
   
  export SVN_PARENT_PATH=$SCM_FOLD/repos

  ## just needed by svn.bash
  SVN_APACHE2_CONF=$APACHE2_LOCAL/svn.conf

  ## these are needed by both SVN + Trac  
  SVN_APACHE2_AUTH=$APACHE2_LOCAL/svn-apache2-auth
  SVN_APACHE2_AUTHZACCESS=$APACHE2_LOCAL/svn-apache2-authzaccess	

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
	local base=${2:-/tmp/env/$FUNCNAME}
	
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

  local msg="=== $FUNCNAME :"
  local access=${1:-formlogin}
  
  [ -z $APACHE2_BASE ] && echo $msg ABORT no APACHE2_BASE && return 1
  
  svn-apache2-conf- $access $APACHE2_BASE 

 ## the files are already Included curtesy of heprez-/apache-/apache-conf-heprez
 ## apache2-conf-connect $SVN_APACHE2_CONF
 ## apache2-conf-connect $TRAC_APACHE2_CONF
 
  svn-apache2-xslt-write $APACHE2_XSLT
  
  local eggcache=$TRAC_EGG_CACHE
  mkdir -p  $eggcache
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
  $ASUDO bash -lc "env- ; svn- ; svn-apache2- ; svn-apache2-authzaccess $* >  $authzaccess"
  $ASUDO chown $APACHE2_USER $authzaccess
   ls -l $authzaccess
   echo =============== cat $authzaccess
  #cat $authzaccess

}







svn-apache2-checkout-test(){

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










