
hepreztrac-propagate(){

   local name=${1:-$SCM_TRAC}
   shift
   echo === hepreztrac-env propagating variables into trac.ini 
   
   tracenv
   tracenv-propagate $name APACHE_LOCAL_FOLDER APACHE_MODE HFAG_PREFIX
 
}