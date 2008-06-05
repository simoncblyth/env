

tracenv-usage(){

cat << EOU
tracenv-propagate  <name>  ENV_NAME_1 ENV_NAME_2 ...        propagate env vars into the trac ini file             
EOU

}



tracenv-propagate(){
 
   local name=${1:-$SCM_TRAC}
   shift
   local tini=$SCM_FOLD/tracs/$name/conf/trac.ini
   
   if [ ! -f $tini ]; then
        echo tracenv-propagate error no such trac.ini file $tini  
        return 1 
   fi
   
   local vars="TRAC_COMMON $*"

   trac-ini-

   for var in $vars
   do 
      eval vval=\$$var
      if [ "X$vval" == "X" ]; then
         echo tracenv-propagate error not defined $var
      else   
         local cmd="trac-ini-edit $tini tracenv:$var:$vval"
         echo $cmd
         eval $cmd 
      fi 
   done
         
}