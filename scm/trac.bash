#
#  prerequisites to trac :
#
#      svn
#      apache2
#      sqlite
#      pysqlite
#      python
#      swig
#      modpython OR modwsgi
#      clearsilver
#
#    
#   usage:
#
#      trac-x
#      trac-i
#
#      trac-perms
#      trac-open
#      trac-authz-check
#      trac-log
#      trac-authz
#
#      trac-apache2-conf  frontend
#
#                                 name: name of repository
#                             frontend: modwsgi OR modpython
#
#
#      trac-webadmin-plugin-get
#      trac-pygments-plugin-get
# 
#      trac-get
#      trac-install
#


export TRAC_NAME=trac-0.10.4
export TRAC_HOME=$LOCAL_BASE/trac
TRAC_APACHE2_CONF=etc/apache2/trac.conf 



trac-x(){ scp $SCM_HOME/trac.bash ${1:-$TARGET_TAG}:$SCM_BASE; }
trac-i(){ . $SCM_HOME/trac.bash ; }


trac-perms(){

  name=${1:-dummy}

  ## remove all permissions from anonymous , but then cannot even login !
  trac-admin $SCM_FOLD/tracs/$name permission remove anonymous '*'
  trac-admin $SCM_FOLD/tracs/$name permission add    anonymous WIKI_VIEW 
  trac-admin $SCM_FOLD/tracs/$name permission add    anonymous BROWSER_VIEW 
  trac-admin $SCM_FOLD/tracs/$name permission add    anonymous LOG_VIEW
  trac-admin $SCM_FOLD/tracs/$name permission add    anonymous FILE_VIEW
  trac-admin $SCM_FOLD/tracs/$name permission add    anonymous CHANGESET_VIEW
  
  trac-admin $SCM_FOLD/tracs/$name permission add    authenticated TICKET_MODIFY
  trac-admin $SCM_FOLD/tracs/$name permission add    blyth TRAC_ADMIN
}


trac-authz-check(){
  find $SCM_FOLD/tracs -name trac.ini -exec grep -H auth {} \;
}


trac-log(){

  name=${1:-dummy}
  cat $SCM_FOLD/tracs/$name/log/trac.log

}


trac-apache2-conf(){
	
   name=${1:-dummy}
   frontend=${2:-modpython}

   ## name not used at the moment, but potentially will have different users per repository so keep it
   #[ "$name" == "dummy" ] && echo need 1st argument corresponsing to repository name  && return


   echo ============== trac-apache2-conf name:$name frontend:$frontend
   conf=$APACHE2_HOME/$TRAC_APACHE2_CONF
   
   echo =============== creating trac config file for apache2 in $conf with userfile $userfile
   userfile=$SVN_APACHE2_AUTH

   if [ "$frontend" == "modwsgi2" ]; then
     $ASUDO modwsgi-tracs-conf2 $userfile  >  $conf
   elif [ "$frontend" == "modwsgi" ]; then	 
     $ASUDO modwsgi-tracs-conf $userfile  >  $conf
   else	 
     $ASUDO modpython-tracs-conf $userfile  >  $conf
   fi 
   
   echo =============== cat $conf
   cat $conf
   
   echo =============== cat \$APACHE2_HOME/$userfile
   cat $APACHE2_HOME/$userfile

   echo =============== connecting the conf file $conf to $APACHE2_CONF if not done already 
   grep $TRAC_APACHE2_CONF $APACHE2_CONF  || $ASUDO echo "Include $TRAC_APACHE2_CONF"  >> $APACHE2_CONF  

   # after this above restart apache2 , and then try trac-open

  [ "$APACHE2_HOME/sbin" == $(dirname $(which apachectl)) ] || (  echo your PATH to apache2 executables is not setup correctly  && return ) 
  apachectl configtest && echo restarting apache2 && apachectl restart || echo apachectl configtest failed

   
}



#
#   python distribution primer ..
#
#      python setup.py   ... is the "standard" ? Distutils way of installing 
#
# [blyth@hfag 0.10]$ python setup.py bdist --help-formats
# List of available distribution formats:
#   --formats=rpm      RPM distribution
#   --formats=gztar    gzip'ed tar file
#   --formats=bztar    bzip2'ed tar file
#	--formats=ztar     compressed tar file
#	--formats=tar      tar file
#	--formats=wininst  Windows executable installer
#	--formats=zip      ZIP file
#	--formats=egg      Python .egg file
#
#

trac-xmlrpc-plugin-get(){

  cd $LOCAL_BASE/trac
  mkdir -p plugins && cd plugins
  svn co http://trac-hacks.org/svn/xmlrpcplugin 

  cd xmlrpcplugin/0.10
  python setup.py install

  cd  $PYTHON_HOME/lib/python2.5/site-packages
  ls -alst TracXMLRPC-0.1-py2.5.egg
  cat easy-install.pth

#  i used the above "install" method that puts the egg into site-packages 
#   ... but http://www.trac-hacks.org/wiki/XmlRpcPlugin
#  suggests the below..   i assume the difference is egg positioning only 
#
# lay an egg ... makes dirs : build, TracXMLRPC.egg-info , dist   
#
#  python setup.py bdist_egg
#  ls -alst dist/TracXMLRPC-0.1-py2.5.egg
#  cp dist/*.egg /srv/trac/env/plugins
# 

}




trac-xmlrpc-plugin-enable(){

 # todo : generalize

   name=${1:-env}
   tini=$SCM_FOLD/tracs/$name/conf/trac.ini
 #  cp -f $tini /tmp/
 #  tini=/tmp/trac.ini

   [ -f "$tini" ] || ( echo trac-enable-component ABORT trac config file $tini not found  && return 1 )

   ## adds 
   grep \\[components\\] $tini && echo components section in $tini already || ( $SUDO printf "\n[components]\n"  >> $tini )
   grep "tracrpc.*"  $tini     && echo already || ( $SUDO printf "tracrpc.* = enabled \n" >> $tini )
  
}



trac-webadmin-plugin-get(){

  cd $LOCAL_BASE/trac
  mkdir -p plugins && cd plugins
  svn co http://svn.edgewall.com/repos/trac/sandbox/webadmin/

  cd webadmin
  python setup.py install

#   rev 5285 on grid1   
#   rev 5324 on hfag
}


trac-pygments-plugin-get(){

## http://trac-hacks.org/wiki/TracPygmentsPluginA
##
##  

   cd $LOCAL_BASE/trac
   mkdir -p plugins && cd plugins
   
   nam=tracpygmentsplugin
   zip=$nam.zip
   test -f $zip || curl -o $zip "http://trac-hacks.org/changeset/latest/tracpygmentsplugin?old_path=/&filename=tracpygmentsplugin&format=zip"

   unzip -l $zip
   test -d $nam || unzip $zip
   
   cd $nam
   cd 0.10
   python setup.py install

# Installed /disk/d4/dayabay/local/python/Python-2.5.1/lib/python2.5/site-packages/TracPygments-0.3dev-py2.5.egg

}



trac-get(){
  
  nam=$TRAC_NAME
  tgz=$nam.tar.gz
  url=http://ftp.edgewall.com/pub/trac/$tgz

  cd $LOCAL_BASE
  test -d trac || ( $SUDO mkdir trac && $SUDO chown $USER trac )
  cd trac  

  test -f $tgz || curl -o $tgz $url
  test -d build || mkdir build
  test -d build/$nam || tar -C build -zxvf $tgz 
}

trac-install(){

  nam=$TRAC_NAME
  cd $LOCAL_BASE/trac/build/$nam

  $PYTHON_HOME/bin/python ./setup.py --help    ## not very helpful
  $PYTHON_HOME/bin/python ./setup.py install --help   ## more helpful
  $PYTHON_HOME/bin/python ./setup.py install 

  #
  # creates and installs into :
  #
  #      $PYTHON_HOME/lib/python2.5/site-packages/trac
  #      $PYTHON_HOME/share/trac/templates
  #      $PYTHON_HOME/share/trac/htdocs
  #
  #      $PYTHON_HOME/bin/trac-admin
  #      $PYTHON_HOME/bin/tracd
  #
}



