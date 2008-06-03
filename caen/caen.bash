
caen-usage(){

cat << EOU

   caen-blahblah

   



EOU


}


caen-env(){

   elocal-

   if [ "$NODE_TAG" == "HKVME" ]; then
     export CAEN_CD=/blah/blah
     export LD_LIBRARY_PATH=$CAEN_CD:$LD_LIBRARY_PATH
   fi

}




