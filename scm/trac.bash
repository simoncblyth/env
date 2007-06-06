#
#  TODO:
#
#      To submit tickets or edit wiki pages, you need to register an account and log in. The 'Register' link can be found at the top of the page.
#
#
#
#
#
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



trac-setup-perms(){

    local name=${1:-$SCM_TRAC}
    local level=${2:-$SCM_SECURITY_LEVEL}

	views="WIKI_VIEW TICKET_VIEW BROWSER_VIEW LOG_VIEW FILE_VIEW CHANGESET_VIEW MILESTONE_VIEW ROADMAP_VIEW REPORT_VIEW"	 
    other="TIMELINE_VIEW SEARCH_VIEW"
	hmmm="CONFIG_VIEW"
    wiki="WIKI_CREATE WIKI_MODIFY"
	ticket="TICKET_CREATE TICKET_APPEND TICKET_CHGPROP TICKET_MODIFY"
    milestone="MILESTONE_CREATE MILESTONE_MODIFY"
    report="REPORT_SQL_VIEW REPORT_CREATE REPORT_MODIFY"
 
 
    ## remove WIKI_DELETE MILESTONE_DELETE REPORT_DELETE ... leave those to admin only
    ## allow unauth to REPORT_VIEW 
 
    if [ "$level" == "loose" ]; then
 
      trac-user-perms $name anonymous     "$views $other" 
	  trac-user-perms $name authenticated "$views $other $hmmm $wiki $ticket $milestone $report"
      trac-user-perms $name admin TRAC_ADMIN
 
    elif [ "$level" == "tight" ]; then
    
      trac-user-perms $name anonymous     "$views $other" 
	  trac-user-perms $name authenticated "$views $other $hmmm $wiki $ticket $milestone $report"
      trac-user-perms $name admin TRAC_ADMIN 
      
    else
        echo "ERROR security level $level is no implemented "
    fi            
       
             
      
    ## does TRAC_ADMIN include XML_RPC ? yes 

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






#
#   python distribution primer ..
#
#      python setup.py   ... is the "standard" ? Distutils way of installing 
#
# [blyth@hfag 0.10]$ python setup.py bdist --help-formats
# List of available distribution formats:
#   --formats=rpm      RPM distribution
#   --formats=gztar    gziped tar file
#   --formats=bztar    bzip2ed tar file
#	--formats=ztar     compressed tar file
#	--formats=tar      tar file
#	--formats=wininst  Windows executable installer
#	--formats=zip      ZIP file
#	--formats=egg      Python .egg file
#
#







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















trac-ini(){
  local name=${1:-$SCM_TRAC}
  $SUDO vi  $SCM_FOLD/tracs/$name/conf/trac.ini
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

trac-user-perms(){

   name=${1:-$SCM_TRAC}
   user=${2:-anonymous}
   shift 
   shift 
   
   ## remove all permissions first ... and then apply 

   trac-permission $name remove $user  \'*\'
   
   #for perm in $*
   #do	   
   #   trac-permission $name add $user $perm
   #done
   
   trac-permission $name add $user $*
}


