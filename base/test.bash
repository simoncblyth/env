test-cp(){

   cd /tmp 
   local f=100m
   [ -f $f ] || mkfile $f $f
   
   rm -f $f.1
   time cp $f $f.1
   
   


}