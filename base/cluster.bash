
cluster-nodes(){
   condor_status | perl -n -e 'BEGIN{ @n=(); }; m/(albert\d*)/ && do { push(@n,$1) if(grep($1 eq $_,@n)==0); } ; END{ print "@n " } '
}

cluster-disks(){

   echo "hello"
}