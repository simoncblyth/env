trachttpauth-vi(){ vi $BASH_SOURCE ; }
trachttpauth-env(){ elocal- ; trac- ; }
trachttpauth-usage(){ cat << EOU

http://trac-hacks.org/wiki/HttpAuthPlugin

Workaround interference between XMLRPCPlugin and AccountManagerPlugin
needed for authenticated xmlrpc access while form based logins in use"


install setuptools subversion issue
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Setuptools reads internal subversion files. As of subversion 1.7 the internals are changed such as
to break setuptools.

  * http://stackoverflow.com/questions/9851911/unrecognized-svn-entries-format-using-buildout
  * http://pypi.python.org/pypi/setuptools_subversion

The reason for this old issue to surface is that on G are using the system python for Trac 
(due to its comparitive stability cf the macports python)

::

    sudo easy_install setuptools_subversion

This did not succeed, probably should update setuptools from 0.6c7
but first lets try pip

::

       sudo easy_install pip

::

	g4pb-2:httpauth blyth$ sudo pip install .
	Unpacking /usr/local/env/otrac/plugins/httpauth
	Running setup.py egg_info for package from file:///usr/local/env/otrac/plugins/httpauth
	unrecognized .svn/entries format in
	Requirement already satisfied (use --upgrade to upgrade): TracAccountManager in /Library/Python/2.5/site-packages/TracAccountManager-0.2.1dev_r3734-py2.5.egg (from TracHTTPAuth==1.1)
	Installing collected packages: TracHTTPAuth
	Running setup.py install for TracHTTPAuth
	unrecognized .svn/entries format in
	Successfully installed TracHTTPAuth
	Cleaning up...



EOU
}


trachttpauth-dir(){ echo $LOCAL_BASE/env/otrac/plugins/httpauth ; }
trachttpauth-cd(){  cd $(trachttpauth-dir) ; }
trachttpauth-get(){
  local dir=$(trachttpauth-dir) 
  mkdir -p $(dirname $dir) && cd $(dirname $dir)
  svn co http://trac-hacks.org/svn/httpauthplugin/trunk $(basename $dir)
}

trachttpauth-install(){
  trachttpauth-cd
  #sudo easy_install -Z -U $PWD
  sudo pip install .
}

trachttpauth-enable(){
   TRAC_INSTANCE=${1:-$TRAC_INSTANCE} trac-configure components:httpauth.\*:enabled
   TRAC_INSTANCE=${1:-$TRAC_INSTANCE} trac-configure httpauth:paths:/xmlrpc,/login/xmlrpc
}






trachttpauth-old-install(){

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

