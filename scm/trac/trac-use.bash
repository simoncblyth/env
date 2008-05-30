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



trac-use-env(){

   elocal-
}



trac-use-usage(){
cat << EOU

trac-use-authz <name> <authz_file>                  edit the ini file for tracitory setting the authz_file
init-edit      <path>  block1:var1:value1 ...       edit ini file based on arguments 
trac-use-admin <name>  <...>                        invoke trac-admin tool          
                           

EOU

}


trac-use-authz(){

   local name=${1:-dummy}
   [ "$name" == "dummy" ] && echo must specify the repo/trac name as the single argument && return 

   local authzaccess=${2:-dummy}
   [ "$authzaccess" == "dummy" ] && echo must give path to apache2 config file setting up authz && return 

   tini=$SCM_FOLD/tracs/$name/conf/trac.ini
   [ -f "$tini" ] || ( echo ERROR $tini does not exist && return 1 )     

   local conf="logging:log_level:INFO logging:log_type:file trac:authz_file:$authzaccess trac:authz_module_name:$name"
   
   echo === trac-use-authz ==== invoke: ini-edit $tini $conf
   type ini-edit 
   ini-edit  $tini $conf   
}


#  
#alias ini-edit="sudo -u $APACHE2_USER $ENV_HOME/base/ini-edit.pl" 




trac-use-admin(){

   local name=${1:-dummy}
   shift
   local tracdir=$SCM_FOLD/tracs/$name
   test -d $tracdir || ( echo tracdir $tracdir does not exist && return 1 )

   local group=$(id -gn)

   local cmd="sudo chown -R $USER:$group $tracdir"
   echo $cmd
   eval $cmd
   
   trac-admin $tracdir $*
   sudo chown -R $APACHE2_USER:$APACHE2_USER $tracdir
}


trac-use-authz-old(){

   name=${1:-dummy}
   [ "$name" == "dummy" ] && echo must specify the repo/trac name as the single argument && return 

   ## tweak the trac.ini for fineGrainedPermissions
   ## http://trac.edgewall.org/wiki/FineGrainedPermissions
   ##
   tini=$SCM_FOLD/tracs/$name/conf/trac.ini
   authzaccess="$APACHE2_BASE/$SVN_APACHE2_AUTHZACCESS"


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


