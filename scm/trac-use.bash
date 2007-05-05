

trac-use-authz(){

   name=${1:-dummy}
   [ "$name" == "dummy" ] && echo must specify the repo/trac name as the single argument && return 

   ## tweak the trac.ini for fineGrainedPermissions
   ## http://trac.edgewall.org/wiki/FineGrainedPermissions
   ##
   tini=$SCM_FOLD/tracs/$name/conf/trac.ini
   authzaccess="$APACHE2_HOME/$SVN_APACHE2_AUTHZACCESS"
   
   echo ================== tweaking $tini inserting authzaccess:$authzaccess and name:$name 
   cp -f $tini $tini.orig
   perl -pi -e "s|^(authz_file =).*\$|\$1 $authzaccess|" $tini
   perl -pi -e "s|^(authz_module_name =).*\$|\$1 $name|" $tini
   perl -pi -e "s|^(log_type =).*\$|\$1 file|" $tini
   diff $tini{.orig,}

}


