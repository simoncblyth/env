
trac-plugin-all-usage(){ cat << EOU

EOU
}

trac-plugin-all-enable(){

   local name=${1:-$ENVBASE}
   
   trac-plugin-restrictedarea-conf $name 
   trac-plugin-accountmanager-conf $name


}
