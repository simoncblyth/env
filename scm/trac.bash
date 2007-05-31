#
#  TODO:
#    integrate the wiki backup and restore scripts
#    modify backup folder to handle multiple repositories
#
#
#  debugging the fatal python error...
#
#     python -v $(which trac-admin)
#     python -v $(which trac-admin)  /var/scm/tracs/test initenv test sqlite:db/trac.db svn /var/scm/repos/test /usr/local/python/Python-2.5.1/share/trac/templates
#
#   
#      python -vc "import libsvn.fs"
#
#
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
#      trac-setup-perms      assigns permissions to :  anonymous, authenticated and admin users 
#         trac-permission    set a single permission 
#         trac-user-perms    assign all the permissions for a user , starts by wiping any preexisting permissions
#
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
#           create the users file with "scm-add-user name" before doing this 
#           do "svn-apache2-settings" to add the three modules to apache..
#
#
#
#      trac-xmlrpc-wiki-backup  [pagenames]
#      trac-xmlrpc-wiki-restore [pagenames]
#
#    eg   trac-xmlrpc-wiki-backup  WikiStart OtherPage      get the page(s) from the remote server to local $SCM_FOLD/wiki-backup
#         trac-xmlrpc-wiki-restore WikiStart                put the page(s) from local $SCM_FOLD/wiki-backup to remote server
#
#             -   without arguments defaults to all wiki pages...
#             -   allows local wiki editing
#
#
#      trac-webadmin-plugin-get
#      trac-pygments-plugin-get
# 
#      trac-get
#      trac-install
#
#      trac-xmlrpc-plugin-permission
#


export TRAC_NAME=trac-0.10.4
TRAC_NIK=trac

export TRAC_HOME=$LOCAL_BASE/$TRAC_NIK

TRAC_APACHE2_CONF=etc/apache2/trac.conf 
export TRAC_ENV_XMLRPC="http://$USER:$NON_SECURE_PASS@$SCM_HOST:$SCM_PORT/tracs/$SCM_TRAC/login/xmlrpc"

export TRAC_SHARE_FOLD=$PYTHON_HOME/share/trac



trac-x(){ scp $SCM_HOME/trac.bash ${1:-$TARGET_TAG}:$SCM_BASE; }
trac-i(){ . $SCM_HOME/trac.bash ; }

trac-permission(){
   local name=${1:-$SCM_TRAC}
   shift
   echo $SUDO trac-admin $SCM_FOLD/tracs/$name permission $*
        $SUDO trac-admin $SCM_FOLD/tracs/$name permission $*
}


trac-user-perms(){

   name=${1:-$SCM_TRAC}
   user=${2:-anonymous}
   shift 
   shift 
   
   ## remove all permissions first ... and then apply 

   trac-permission $name remove $user  \'*\'
   for perm in $*
   do	   
      trac-permission $name add $user $perm
   done
}

trac-setup-perms(){

    name=${1:-$SCM_TRAC}

	views="WIKI_VIEW TICKET_VIEW BROWSER_VIEW LOG_VIEW FILE_VIEW CHANGESET_VIEW MILESTONE_VIEW ROADMAP_VIEW"	 
    other="TIMELINE_VIEW SEARCH_VIEW"
	hmmm="CONFIG_VIEW"
    wiki="WIKI_CREATE WIKI_MODIFY"
	ticket="TICKET_CREATE TICKET_APPEND TICKET_CHGPROP TICKET_MODIFY"
    milestone="MILESTONE_CREATE MILESTONE_MODIFY"
    report="REPORT_VIEW REPORT_SQL_VIEW REPORT_CREATE REPORT_MODIFY"
 
 
    ## remove WIKI_DELETE MILESTONE_DELETE REPORT_DELETE ... leave those to admin only
 
    trac-user-perms $name anonymous     "$views $other" 
	trac-user-perms $name authenticated "$views $other $hmmm $wiki $ticket $milestone $report"
    trac-user-perms $name admin TRAC_ADMIN 
   
   ## does TRAC_ADMIN include XML_RPC
}



trac-components(){ 
   local name=${1:-$SCM_TRAC}
   shift
   echo sudo trac-admin $SCM_FOLD/tracs/$name component $*
        sudo trac-admin $SCM_FOLD/tracs/$name component $*
}

trac-setup-components(){

    local name=${1:-$SCM_TRAC}
    local pairs="red:admin green:admin blue:admin"

    ## remove the default components 
    trac-components $name remove component1
    trac-components $name remove component2

    for pair in $pairs
    do
	    local component=$(echo $pair | cut -f1 -d:)
	    local     owner=$(echo $pair | cut -f2 -d:)
        trac-components $name add $component $owner
    done	   
    trac-components $name list 
}



trac-conf-notification(){
   echo simply make the following settings ... to get emails on ticket changes
   echo the user settings should include the email address ... if doesnt correspond to default domain
cat << EOX
smtp_default_domain = hep1.phys.ntu.edu.tw
smtp_enabled = true
always_notify_owner = true
always_notify_reporter = true
always_notify_updater = true
EOX
   echo sudo vi $SCM_FOLD/tracs/env/conf/trac.ini
}


trac-authz-check(){
  find $SCM_FOLD/tracs -name trac.ini -exec grep -H auth {} \;
}


trac-log(){

  name=${1:-$SCM_TRAC}
  cat $SCM_FOLD/tracs/$name/log/trac.log
  ls -alst  $SCM_FOLD/tracs/$name/log/trac.log

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
     $ASUDO bash -lc "modwsgi-tracs-conf2 $userfile  >  $conf "
   elif [ "$frontend" == "modwsgi" ]; then	 
     $ASUDO bash -lc "modwsgi-tracs-conf $userfile  >  $conf "
   else	 
     $ASUDO bash -lc "modpython-tracs-conf $userfile  >  $conf "
   fi 
   
   echo =============== cat $conf
   cat $conf
   
   echo =============== cat \$APACHE2_HOME/$userfile
   cat $APACHE2_HOME/$userfile

   echo =============== connecting the conf file $conf to $APACHE2_CONF if not done already 
   grep $TRAC_APACHE2_CONF $APACHE2_CONF  || $ASUDO bash -c "echo \"Include $TRAC_APACHE2_CONF\"  >> $APACHE2_CONF "  

   # after this above restart apache2 , and then try trac-open

  [ "$APACHE2_HOME/sbin" == $(dirname $(which apachectl)) ] || (  echo your PATH to apache2 executables is not setup correctly  && return ) 
  apachectl configtest && echo restarting apache2 && $ASUDO apachectl restart || echo apachectl configtest failed

   
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


#
#  backup and restore of wiki pages via xmlrpc 
#     - currently does all pages 
#  
#
#  somehow getting permission error ... the log/trac.log becomes owned by root
#



trac-wiki-backup(){
  
  local name=${1:-$SCM_TRAC}
  shift
  
  cd $SCM_FOLD
  [ -d backup ] || ( sudo mkdir backup && sudo chown $USER backup )
  
  dir="backup/tracs/$SCM_HOST/$name/wiki"
  [ -d $dir ] || ( mkdir -p $dir || ( echo abort && return 1 ))
  cd $dir  
  python $HOME/$SCM_BASE/xmlrpc-wiki-backup.py $name $*
}

trac-wiki-restore(){
  
  local name=${1:-$SCM_TRAC}
  shift
  
  cd $SCM_FOLD
  dir="backup/tracs/$SCM_HOST/$name/wiki"
  [ -d $dir ] || ( echo abort ... must backup before can restore  && return 1 )
  cd $dir
  python $HOME/$SCM_BASE/xmlrpc-wiki-restore.py $name $*
}


trac-xmlrpc-plugin-test(){

  cd  /tmp && mkdir -p tractest && cd tractest
  python $HOME/$SCM_BASE/xmlrpc-wiki-backup.py $*

}

trac-xmlrpc-plugin-get(){


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


trac-xmlrpc-prepare(){

   name=${1:-$SCM_TRAC}

   trac-xmlrpc-plugin-install $name 
   trac-xmlrpc-plugin-enable  $name
   trac-xmlrpc-plugin-permission $name

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


trac-xmlrpc-plugin-install(){

  name=${1:-$SCM_TRAC}

  egg=TracXMLRPC-0.1-py2.5.egg

  
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





trac-plugin-enable(){

   ## globally installed plugins need to be enabled ..

   name=${1:-$SCM_TRAC}
   plugin=${2:-dummy}
   
   tini=$SCM_FOLD/tracs/$name/conf/trac.ini

   [ -f "$tini" ] || ( echo trac-enable-component ABORT trac config file $tini not found  && return 1 )

   ## adds compenents section if not there already and appends some config ...
   
   grep \\[components\\] $tini && echo components section in $tini already || ( sudo bash -c "echo \"[components]\"         >> $tini " )
   grep "$plugin.*"      $tini && echo already                             || ( sudo bash -c "echo \"$plugin.* = enabled \" >> $tini " )

   cat $tini

   ## NB the "sudo bash -c" construct is in order for the redirection to be done with root privilege
}



trac-xmlrpc-plugin-enable(){
   
   name=${1:-$SCM_TRAC}
   trac-plugin-enable $name tracrpc
   
}



trac-xmlrpc-plugin-permission(){

   name=${1:-$SCM_TRAC}
   sudo trac-admin $SCM_FOLD/tracs/$name permission add blyth XML_RPC
   sudo trac-admin $SCM_FOLD/tracs/$name permission list 

## thence 

}


trac-xmlrpc-open(){

   name=${1:-$SCM_TRAC}
   open http://$USER:$NON_SECURE_PASS@$SCM_HOST:$SCM_PORT/tracs/$name/login/xmlrpc
}


trac-plugin-webadmin-get(){

  cd $LOCAL_BASE/trac
  mkdir -p plugins && cd plugins
  svn co http://svn.edgewall.com/repos/trac/sandbox/webadmin/


}

trac-plugin-webadmin-install(){

  cd $LOCAL_BASE/trac/plugins 
  cd webadmin
  python setup.py install

#   rev 5285 on grid1   
#   rev 5324 on hfag  
# huh ... nowhere to be found 

}

trac-plugin-webadmin-enable(){
   
   name=${1:-$SCM_TRAC}
   trac-plugin-enable $name webadmin 
   
}




trac-plugin-tracnav-get(){

   #  documented at 
   # http://svn.ipd.uka.de/trac/javaparty/wiki/TracNav
   #

   cd $LOCAL_BASE/trac
   [ -d "plugins" ] || mkdir -p plugins
   cd plugins
    
   svn co http://svn.ipd.uka.de/repos/javaparty/JP/trac/plugins/tracnav/
   cd tracnav

}

trac-plugin-tracnav-install(){

    cd $LOCAL_BASE/trac/plugins || ( echo error no plugins folder && return 1 ) 
    cd tracnav
    python setup.py install 
   
  ##  Installed /data/usr/local/python/Python-2.5.1/lib/python2.5/site-packages/TracNav-3.92-py2.5.egg
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


trac-site-wipe(){

  cd $PYTHON_SITE && sudo rm -rf trac && sudo rm -rf trac*.egg-info && ls -alst $PYTHON_SITE 
}

trac-wipe(){

   nam=$TRAC_NAME
   nik=$TRAC_NIK
   cd $LOCAL_BASE/$nik
   rm -rf build/$nam
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


