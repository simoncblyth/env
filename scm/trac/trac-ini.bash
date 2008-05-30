

trac-ini-usage(){

   cat << EOU
   

   trac-ini-edit /path/to/trac.ini <triplets...>
   
   
EOU
}


trac-ini-env(){
  elocal-
}


init-edit(){
   local msg="=== $FUNCNAME :"
   echo $msg shim for backward compatibility ... change to trac-ini-edit
   
   trac-ini-edit $*
}


trac-ini-edit(){

   local path=$1
   shift
 
   sudo perl $ENV_HOME/base/ini-edit.pl $path $*  
#
#  
#   on hfag machine ... skipping the perl causes error... 
#   /usr/bin/env: perl -w: No such file or directory
#
#
#


}

trac-ini-edit-prior(){

    
   # 
   # utility for editing INI files ... moved from base/file.bash as needs APACHE2_USER
   # note the APACHE2_USER is very limited in capabilities , so dont try to do the editing as it ...
   # just hand over ownership as the last step
   #
   #   moved from trac-conf.bash  as need it remotely, without terminal attached
   # 
    
    
   local path=$1
   shift 
   local pmpath=$ENV_HOME/base/INI.pm 
   sudo perl -e 'require "$ENV{'ENV_HOME'}/base/INI.pm" ; &INI::EDIT(@ARGV) ; ' $path $*
   sudo chown $APACHE2_USER:$APACHE2_USER $path

   
}

