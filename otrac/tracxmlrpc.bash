tracxmlrpc-vi(){  vim $BASH_SOURCE ; }
tracxmlrpc-usage(){ cat << EOU

tracxmlrpc
===========
 
http://www.trac-hacks.org/wiki/XmlRpcPlugin

::

tracxmlrpc-test
             -get
             -prepare
             -install
             -enable
             -permission


tracxmlrpc-open
   WSDL style descripion of the protocol


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
# seems that if you test things too quickly after restart the log file becomes owned by
# "root" ... presumably the request is handled by the primary apache process , prior to it spawning 
# resulting in the rootified trac.log 
# 
#  can fix with : 
#       sudo chown www $SCM_FOLD/tracs/$name/log/trac.log
#
# tracxmlrpc-test    
#


::

	g4pb-2:trunk blyth$ tracxmlrpc-enable
	=== trac-edit-ini : diff /var/scm/tracs/workflow/conf/trac.ini /tmp/env/trac/trac-edit-ini/trac.ini
	30a31
	> tracrpc.* = enabled
	153c154
	< query.target = mainnav   # metanav
	---
	> query.target = mainnav# metanav
	158,159c159
	< docs.target = mainnav   # metanav
	< 
	---
	> docs.target = mainnav# metanav



On G, only after /usr/bin/easy_install (targetting the system python) does:
  
  * the plugin appear in admin, http://localhost/tracs/workflow/admin/general/plugin
  * XML_RPC permission is recognized with tracxmlrpc-permission


::

	Running setup.py -q bdist_egg --dist-dir /usr/local/env/otrac/plugins/xmlrpcplugin/trunk/egg-dist-tmp-h1gK4_
	Adding TracXMLRPC 1.1.2 to easy-install.pth file
	
	Installed /Library/Python/2.5/site-packages/TracXMLRPC-1.1.2-py2.5.egg
	Processing dependencies for TracXMLRPC==1.1.2
	Finished processing dependencies for TracXMLRPC==1.1.2


EOU
}


tracxmlrpc-env(){
   private-
   
   #export TRAC_ENV_XMLRPC="http://$USER:$(private-val NON_SECURE_PASS)@$SCM_HOST:$SCM_PORT/tracs/$SCM_TRAC/login/xmlrpc"
   export TRAC_ENV_XMLRPC="http://$USER:$(private-val NON_SECURE_PASS)@localhost/tracs/$TRAC_INSTANCE/login/xmlrpc"
}


tracxmlrpc-test(){
  local tmp=/tmp/env/$FUNCNAME
  mkdir -p $tmp && cd $tmp
  python $(env-home)/otrac/xmlrpc-wiki-backup.py $*
}

tracxmlrpc-dir(){ echo $LOCAL_BASE/env/otrac/plugins/xmlrpcplugin ; }
tracxmlrpc-cd(){  cd $(tracxmlrpc-dir)/$1 ; }
tracxmlrpc-get(){
  local iwd=$PWD
  local dir=$(tracxmlrpc-dir)
  mkdir -p $(dirname $dir) && cd $(dirname $dir)
  svn co http://trac-hacks.org/svn/xmlrpcplugin 
  #cd $PWD
}

tracxmlrpc-prepare(){

   local name=${1:-$TRAC_INSTANCE}

   [ -z "$name" ] && echo $msg TRAC_INSTANCE must be defined or named as argument && return 1 

   tracxmlrpc-install 
   tracxmlrpc-enable  $name
   tracxmlrpc-permission $name
}


tracxmlrpc-install(){

  #local name=${1:-$SCM_TRAC}
  #local egg=TracXMLRPC-0.1-py2.5.egg
  local egg=TracXMLRPC-1.1.2-py2.5.egg

  #local plugd
  #
  #if [ "$name" == "global" ]; then
  #   plugd=$TRAC_SHARE_FOLD/plugins
  #else	  
  #   plugd=$SCM_FOLD/tracs/$name/plugins
  #fi
  #
  #if [ -d "$plugins_dir/$egg" ]; then
  #	 echo the plugin is already present in $plugins_dir/$egg
  #	 ls -alst $plugins_dir
  #	 ls -alst $plugins_dir/$egg
  #else

     #cd $LOCAL_BASE/trac/plugins/xmlrpcplugin/0.10


     tracxmlrpc-cd trunk
     
     # sudo python setup.py install  NOPE, macports python the wrong one ?
     # ls -alst $(python-site)/$egg

     sudo easy_install -Z -U $PWD

     #sudo cp dist/$egg $plugins_dir/
     #cd $plugins_dir && python-crack-egg  $egg    ## convert the egg file into a folder
  #fi
}


tracxmlrpc-enable(){
   #local name=${1:-$TRAC_INSTANCE}
   #ini-edit $SCM_FOLD/tracs/$name/conf/trac.ini components:tracrpc.\*:enabled
   TRAC_INSTANCE=${1:-$TRAC_INSTANCE} trac-configure components:tracrpc.\*:enabled
}


tracxmlrpc-permission(){
   local name=${1:-$TRAC_INSTANCE}
   sudo trac-admin $SCM_FOLD/tracs/$name permission add blyth XML_RPC
   sudo trac-admin $SCM_FOLD/tracs/$name permission list 
}



tracxmlrpc-logurl(){
   private-
   echo http://$USER:$(private-val NON_SECURE_PASS)@localhost/tracs/${1:-$TRAC_INSTANCE}/login/xmlrpc
}

tracxmlrpc-open(){
   open $(tracxmlrpc-logurl)
}





