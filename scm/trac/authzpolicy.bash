

authzpolicy-usage(){

cat << EOU

     http://trac.edgewall.org/wiki/TracFineGrainedPermissions


authzpolicy-prepare    <name> <path-to-policy-file>
            get and install the authz_policy plugin into the <name> trac env and 
			configure the env and place the policy file by invoking the below

authzpolicy-get  <name>
authzpolicy-conf <name> <path-to-policy-file>


   see trac-conf.bash for the CoarseGrained policy setup that this hones 


EOU

}


authzpolicy-env(){

   heprez-
   apache-
   [ -z $APACHE__LOCAL ] && echo $msg APACHE__LOCAL must be defined && return 1 

   export AUTHZPOLICY_NAME=authz_policy.py
   export AUTHZPOLICY_CONF=authz_policy.conf

}

authzpolicy-prepare(){

    authzpolicy-get  $*
	authzpolicy-conf $*
		
}  

authzpolicy-get(){

   local name=${1:-$SCM_TRAC}
   
   local msg="=== $FUNCNAME :"
   local tmp=/tmp/$FUNCNAME && mkdir -p $tmp
   local iwd=$PWD
 
   
   local nam=$AUTHZPOLICY_NAME
   
   local plugins=$SCM_FOLD/tracs/$name/plugins
   local tgt=$plugins/$nam
   [ -f $tgt ] &&  echo $msg tgt $tgt already placed && return 1
   
   
	cd $tmp
   
   local url=http://svn.edgewall.com/repos/trac/trunk/sample-plugins/permissions/$nam
   [ -f $nam ] || svn export $url 
   
   echo $msg placing tgt $tgt
   sudo -u $APACHE2_USER cp $nam $tgt 
   
   
   
   # an example
   # local urc=http://swapoff.org/files/authzpolicy.conf
   # [ -f authzpolicy.conf ] || curl -O $urc 

   
   cd $iwd
}




authzpolicy-conf(){

  local msg="=== $FUNCNAME :"
  local name=${1:-$SCM_TRAC}
  
  local def
  case $name in
    workflow) def=$WORKFLOW_HOME/wapache/authz_policy.conf  ;; 
           *) echo $msg not default for name $name ;;
  esac  		   
 
  local policy=${2:-$def}
  
  [ ! -f $policy ] && echo $msg no such policy file $policy && return  1
  
  local iwd=$PWD
  local conf=$SCM_FOLD/tracs/$name/conf
  echo $msg in $conf 
  cd $conf || return 1
  
  [ ! -f trac.ini ] && echo $msg no trac.ini in $PWD && return 1
  
  local authzpolicy=$PWD/$AUTHZPOLICY_CONF
   
  local cmd=$(cat << EOC 
       ini-edit trac.ini 
	        trac:permission_policies:AuthzPolicy,DefaultPermissionPolicy 
			authz_policy:authz_file:$authzpolicy
            components:authz_policy:enabled 
EOC)

   echo $cmd
   eval $cmd
   
   echo $msg copying policy $policy into $conf	     
   sudo -u $APACHE2_USER cp $policy $AUTHZPOLICY_CONF 
	  
   cd $iwd	  
	  
}