tracxmlrpc-vi(){  vim $BASH_SOURCE ; }
tracxmlrpc-usage(){ cat << EOU

tracxmlrpc
===========
 
http://www.trac-hacks.org/wiki/XmlRpcPlugin

tracxmlrpc-open
   WSDL style description of the protocol

C2 Installation
----------------

Relevant python misses TracXMLRPC and TracHTTPAuth on C2::

    C2: /data/env/system/python/Python-2.5.6/lib/python2.5/site-packages
    G:  /Library/Python/2.5/site-packages/


#. manually added egg path to easy-install.pth, after::

    python setup.py bdist_egg --dist-dir $(python-site)

#. Added permission thru web admin interface subject:"blyth" action:"XML_RPC" 

   * http://dayabay.phys.ntu.edu.tw/tracs/env/admin/general/perm 


G Config
-----------


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


On C2, when not using easy_install had to manually egg path to easy-install.pth



::

	Running setup.py -q bdist_egg --dist-dir /usr/local/env/otrac/plugins/xmlrpcplugin/trunk/egg-dist-tmp-h1gK4_
	Adding TracXMLRPC 1.1.2 to easy-install.pth file
	
	Installed /Library/Python/2.5/site-packages/TracXMLRPC-1.1.2-py2.5.egg
	Processing dependencies for TracXMLRPC==1.1.2
	Finished processing dependencies for TracXMLRPC==1.1.2


Test is giving permission denied.  This issue is mentioned in http://trac-hacks.org/wiki/XmlRpcPlugin
as interfence with http://trac-hacks.org/wiki/AccountManagerPlugin 
With a workaround in the form of. 

http://trac-hacks.org/wiki/HttpAuthPlugin



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

   tracxmlrpc-cd trunk
   if [ "$NODE_TAG" == "C2" ]; then 
      python- source
      which python
   fi
   local site=$(python-site) 
   [ "$site" == "" ] && echo $msg ERROR NEED python-site && return 
   python setup.py bdist_egg --dist-dir $site

   echo HAD TO MANUALLY ADD EGG TO easy-install.pth



}


tracxmlrpc-install-old(){

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





