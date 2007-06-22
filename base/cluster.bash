
cluster-nodes(){
   condor_status | perl -n -e 'BEGIN{ @n=(); }; m/(albert\d*)/ && do { push(@n,$1) if(grep($1 eq $_,@n)==0); } ; END{ print "@n " } '
}

cluster-disks(){

   for node in $(cluster-nodes)
   do
      echo ========= $node ========================
      
      if [ "X$node" == "Xalbert11" ]; then
         echo skip 
      else   
         ssh $node "ls -alst /disk/*"
      fi    
   done
}

cluster-cmd(){
  
  for node in $(cluster-nodes)
   do
      echo ========= $node ========================
      
      if [ "X$node" == "Xalbert11" ]; then
         echo skip 
      else   
         ssh $node "bash -lc \"$*\""
      fi    
   done
}