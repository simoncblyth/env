

dybsvn-usage(){

  cat << EOU
  
     Taking a black box view of the repository for bitten usage...
  
         DYBSVN_HOME :  $DYBSVN_HOME
         dybsvn-home : $(dybsvn-home)
  
EOU

}

dybsvn-env(){
   elocal-
   export DYBSVN_HOME=$(dybsvn-home)
}

dybsvn-home(){
   case ${1:-$NODE_TAG} in 
      P) echo /disk/d3/dayabay/local/dyb/trunk_dbg/NuWa-trunk ;; 
      *) echo /tmp ;;
   esac
}
