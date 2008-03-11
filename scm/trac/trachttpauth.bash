

trachttpauth-get(){

  local msg=" === $FUNCNAME needed for authenticated xmlrpc access while form based logins in use"
  echo $msg 

  # http://trac-hacks.swapoff.org/wiki/HttpAuthPlugin 

  cd /tmp
  svn co http://trac-hacks.org/svn/httpauthplugin/0.10
  
  cd 0.10


  

}


trachttpauth-install(){

  local name=${1:-$SCM_TRAC}
  local egg=TracHTTPAuth-1.1-py2.5.egg
  
  if [ "$name" == "global" ]; then
     plugins_dir=$TRAC_SHARE_FOLD/plugins
  else	  
     plugins_dir=$SCM_FOLD/tracs/$name/plugins
  fi
  
  if [ -d "$plugins_dir/$egg" ]; then
	 echo the plugin is already present in $plugins_dir/$egg
	 ls -alst $plugins_dir
	 ls -alst $plugins_dir/$egg
  else
     
	 test -f setup.py || return 1 
	 
     python setup.py bdist_egg
     ls -alst dist/$egg
     sudo cp dist/$egg $plugins_dir/
     cd $plugins_dir && python-crack-egg  $egg    ## convert the egg file into a folder
  
  fi
}
