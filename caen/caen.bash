
caen-usage(){

cat << EOU

   The fuction is used for setting enviroment variables of the VME local mechine user, who
   doesn't have enough authority and want to test the VME software.
   The enviroment vaviables have been setting after "caen-"


EOU


}


caen-env(){

   elocal-

   if [ "$NODE_TAG" == "HKVME" ]; then
     export CAEN_CD=/share/CAEN/V1718/SDK_Ver_3.1/VME_Bridge_Demo_and_Lib/Linux/lib
     export LD_LIBRARY_PATH=$CAEN_CD:$LD_LIBRARY_PATH
   fi


}




