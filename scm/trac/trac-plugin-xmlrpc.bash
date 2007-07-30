
#
#   trac-plugin-xmlrpc-test
#                     -get
#                     -prepare
#                     -install
#                     -enable
#                     -permission
#                     -open
#


trac-plugin-xmlrpc-test(){

  cd  /tmp && mkdir -p tractest && cd tractest
  python $HOME/$SCM_BASE/xmlrpc-wiki-backup.py $*

}

trac-plugin-xmlrpc-get(){


  ## http://www.trac-hacks.org/wiki/XmlRpcPlugin

  cd $LOCAL_BASE/trac
  mkdir -p plugins && cd plugins
  svn co http://trac-hacks.org/svn/xmlrpcplugin 

#  cd xmlrpcplugin/0.10
#  python setup.py install
#
#  cd  $PYTHON_HOME/lib/python2.5/site-packages
#  ls -alst TracXMLRPC-0.1-py2.5.egg
#  cat easy-install.pth
#
#  i used the above "install" method that puts the egg into site-packages 
#   ... but http://www.trac-hacks.org/wiki/XmlRpcPlugin
#  suggests the below..   i assume the difference is egg positioning only 
#
#
#  nope get ... 
#      ExtractionError: Can't extract file(s) to egg cache
#   [Errno 13] Permission denied: '/home/blyth/.python-eggs'
#
#
}


trac-plugin-xmlrpc-prepare(){

   name=${1:-$SCM_TRAC}

   trac-plugin-xmlrpc-install $name 
   trac-plugin-xmlrpc-enable  $name
   trac-plugin-xmlrpc-permission $name

    #
    # echo sleeping a while, prior to doing the test 
	# sleep 10
    #
    # seems that if you test things too quickly after restart the log file becomes owned by
	# "root" ... presumably the request is handled by the primary apache process , prior to it spawning 
    # resulting in the rootified trac.log 
    # 
	#  can fix with : 
    #       sudo chown www $SCM_FOLD/tracs/$name/log/trac.log
    #
    # trac-xmlrpc-plugin-test    
    #

}


trac-plugin-xmlrpc-install(){

  local name=${1:-$SCM_TRAC}
  local egg=TracXMLRPC-0.1-py2.5.egg
  
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
     cd $LOCAL_BASE/trac/plugins/xmlrpcplugin/0.10
     python setup.py bdist_egg
     ls -alst dist/$egg
     sudo cp dist/$egg $plugins_dir/
     cd $plugins_dir && python-crack-egg  $egg    ## convert the egg file into a folder
  fi
}

trac-plugin-xmlrpc-enable(){
   local name=${1:-$SCM_TRAC}
   trac-plugin-enable $name tracrpc
}

trac-plugin-xmlrpc-permission(){
   local name=${1:-$SCM_TRAC}
   sudo trac-admin $SCM_FOLD/tracs/$name permission add blyth XML_RPC
   sudo trac-admin $SCM_FOLD/tracs/$name permission list 
}

trac-plugin-xmlrpc-open(){
   local name=${1:-$SCM_TRAC}
   open http://$USER:$NON_SECURE_PASS@$SCM_HOST:$SCM_PORT/tracs/$name/login/xmlrpc
}





