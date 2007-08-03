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

 trac_iwd=$(pwd)
 TRAC_BASE=$SCM_BASE/trac
 export TRAC_HOME=$SCM_HOME/$TRAC_NIK
 cd $TRAC_HOME


 [ -r trac-conf.bash ]                   && . trac-conf.bash
      
 ## caution webadmin is a prerequisite to accountmanager      
      
 [ -r trac-plugin-webadmin.bash ]        && . trac-plugin-webadmin.bash       
 [ -r trac-plugin-accountmanager.bash ]  && . trac-plugin-accountmanager.bash 

 [ -r trac-plugin-tracnav.bash ]         && . trac-plugin-tracnav.bash 
 [ -r trac-plugin-restrictedarea.bash ]  && . trac-plugin-restrictedarea.bash
 [ -r trac-plugin-pygments.bash ]        && . trac-plugin-pygments.bash     
    
 [ -r trac-macro-latexformulamacro.bash ] && . trac-macro-latexformulamacro.bash  
 [ -r trac-plugin-reposearch.bash ]       && . trac-plugin-reposearch.bash           
                    
                                    
 [ -r trac-build.bash ]                   && . trac-build.bash          ## depends on clearsilver  

 ## new style... reduce env pollution and startup time 

 silvercity(){ . $TRAC_HOME/silvercity.bash ; }
 docutils(){   . $TRAC_HOME/docutils.bash   ; }
 
 trac2mediawiki(){ . $TRAC_HOME/trac2mediawiki.bash   ; }
 trac2latex(){     . $TRAC_HOME/trac2latex.bash   ; }
 





#[ -r trac-test.bash ]      && . trac-test.bash

 ## caution must exit with initial directory
 cd $trac_iwd
 



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

trac-plugin-enable-deprecated(){

    ##  DEPRECATED APPROACH USE ini-edit NOW 

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



