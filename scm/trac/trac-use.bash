#
#
#    trac-admin
#         list the cmds
#
#    trac-admin /var/scm/tracs/env wiki list
#          list pages
#
#    trac-admin /var/scm/tracs/env wiki export WikiStart
#          dumps to stdout 
#
#
#   cat ~/.ssh/id_dsa.pub | ssh user@server "cat - >> ~/.ssh/authorized_keys""
#
#



trac-use-authz(){

   name=${1:-dummy}
   [ "$name" == "dummy" ] && echo must specify the repo/trac name as the single argument && return 

   tini=$SCM_FOLD/tracs/$name/conf/trac.ini
   [ -f "$tini" ] || ( echo ERROR $tini does not exist && return 1 )     

   authzaccess="$APACHE2_HOME/$SVN_APACHE2_AUTHZACCESS" 
   local conf="logging:log_level:INFO logging:log_type:file trac:authz_file:$authzaccess trac:authz_module_name:$name"
   
   ini-edit  $tini $conf   
}


#  
#alias ini-edit="sudo -u $APACHE2_USER $HOME/$ENV_BASE/base/ini-edit.pl" 

ini-edit(){

   # 
   # utility for editing INI files ... moved from base/file.bash as needs APACHE2_USER
   # note the APACHE2_USER is very limited in capabilities , so dont try to do the editing as it ...
   # just hand over ownership as the last step
   #
   #   moved from trac-conf.bash  as need it remotely, without terminal attached
   #
  
   local path=$1
   shift 
    
   sudo perl -e 'require "$ENV{'HOME'}/$ENV{'ENV_BASE'}/base/INI.pm" ; &INI::EDIT(@ARGV) ; ' $path $*
   sudo chown $APACHE2_USER:$APACHE2_USER $path
   
}





trac-use-authz-old(){

   name=${1:-dummy}
   [ "$name" == "dummy" ] && echo must specify the repo/trac name as the single argument && return 

   ## tweak the trac.ini for fineGrainedPermissions
   ## http://trac.edgewall.org/wiki/FineGrainedPermissions
   ##
   tini=$SCM_FOLD/tracs/$name/conf/trac.ini
   authzaccess="$APACHE2_HOME/$SVN_APACHE2_AUTHZACCESS"


   if [ -f "$tini" ]; then
      echo ================== tweaking $tini inserting authzaccess:$authzaccess and name:$name 
      $TSUDO cp -f $tini $tini.orig
      $TSUDO perl -pi -e "s|^(authz_file =).*\$|\$1 $authzaccess|" $tini
      $TSUDO perl -pi -e "s|^(authz_module_name =).*\$|\$1 $name|" $tini
      $TSUDO perl -pi -e "s|^(log_type =).*\$|\$1 file|" $tini
      diff $tini{.orig,}
   else
      echo ================= trac-use-authz trac is not setup ... cannot access $tini 
   fi
	   

}


