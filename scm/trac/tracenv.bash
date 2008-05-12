

tracenv-usage(){

cat << EOU
tracenv-propagate  <name>  ENV_NAME_1 ENV_NAME_2 ...        propagate env vars into the trac ini file             
EOU

}



tracenv-propagate(){
 
   local name=${1:-$SCM_TRAC}
   shift
   local ini=$SCM_FOLD/tracs/$name/conf/trac.ini
   
   if [ ! -f $ini ]; then
        echo tracenv-propagate error no such trac.ini file $ini  
        return 1 
   fi
   
   local vars="TRAC_COMMON $*"

   for var in $vars
   do 
      eval vval=\$$var
      if [ "X$vval" == "X" ]; then
         echo tracenv-propagate error not defined $var
      else   
         local cmd="ini-edit $SCM_FOLD/tracs/$name/conf/trac.ini tracenv:$var:$vval"
         echo $cmd
         eval $cmd 
      fi 
   done
         
}