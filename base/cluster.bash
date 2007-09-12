
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


cluster-touch-disks(){

   ## touching some disks to test time stamping
   
   dirs="$HOME /tmp /disk/d[3-4]" 
   s=$(date +'%s')
   
   printf "%-20s %d[%s] \n"  "touch-disks  now:"  $s $(fmtime $s)  

   f=stamp$$
   for dir in $dirs
   do
	  cd $dir 
	  touch $f && t=$(stat -c %Y $f) && rm -f $f && printf "%-20s %d[%s]  \n" $dir $t $(fmtime $t)   
   done	   

}
