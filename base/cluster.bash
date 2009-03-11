cluster-src(){    echo base/cluster.bash ; }
cluster-source(){ echo ${BASH_SOURCE:-$ENV_HOME/$(cluster-src)} ; }
cluster-vi(){     vi $(cluster-source) ; }
cluster-srcurl(){ echo $(env-localserver)/repos/env/trunk/$(cluster-src) ; }
cluster-usage(){
   cat << EOU

        cluster-src    : $(cluster-src)
        cluster-source : $(cluster-source)
        cluster-srcurl : $(cluster-srcurl)

     cluster-touch-disks
           touching some disks to test time stamping

EOU

}


cluster-fmtime(){
   perl -MPOSIX -e  "print strftime( '%Y%m%d-%H%M%S' , localtime($1) );" 
}


cluster-env(){
   local msg="=== $FUNCNAME :"
}

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
  cluster-touch-disks- $HOME /tmp /disk/d[3-4]
}

cluster-touch-disks-(){

   local msg="=== $FUNCNAME :"
   local s=$(date +'%s')
   printf "$msg %-20s %d[%s] \n"  "touch-disks  now:"  $s $(cluster-fmtime $s)  
   local f=stamp$$
   local dir
   for dir in $*
   do
	  cd $dir 
	  touch $f && t=$(stat -c %Y $f) && rm -f $f && printf "%-20s %d[%s]  \n" $dir $t $(cluster-fmtime $t)   
   done	   

}
