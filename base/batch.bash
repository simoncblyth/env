

#
#  example of testing...
#      batch-submit jobs/test hello world
#


batch-status(){

  if [ "$BATCH_TYPE" == "condor" ]; then
    condor_q
  elif [ "$BATCH_TYPE" == "SGE" ]; then
    qstat -u $USER
  fi

}


batch-submit(){

   ## this is the entry point for instrumented batch job submission 
   
   local  path=$1    ## relative path that acts as a classification string for the job
   local  name=$2    ## name , eg function or script to be invoked
   
   local stamp=$(batch-timestamp)
   echo =========== batch-submit path:$path name:$name stamp:$stamp 

   shift

   [ "X$path" == "X" ] && echo must provide a relative path for classification && return 
   [ "X$name" == "X" ] && echo must provide a script or function to invoke     && return 
 
   batch-prepfold $path $stamp
   
   ## relative branches including the timestamp
   
   local jobsbranch=$(batch-lookup jobsbranch $path $stamp) 
   local databranch=$(batch-lookup databranch $path $stamp)
      
   echo ============ batch-submit jobsbranch:$jobsbranch databranch:$databranch    
      
   ## getting 
   ##  Error from starter on albert4.hepgrid: Failed to open standard output file '/disk/d4/blyth/jobs/test/20070613-160016/condor-use-test.out': Permission denied (errno 13)<   
   ##cd $jobsbranch
   
   cd $databranch   
   
   local script=$name.batch
   batch-script $path "$@" > $script 
   chmod 755 $script
   
   local cmd 
   if [ "$BATCH_TYPE" == "condor" ] ; then
   
       condor-use-subfile $(pwd) $name $script > $name.condorsub
       cmd="condor_submit  $name.condorsub "
        
   elif [ "$BATCH_TYPE" == "SGE" ]; then

       cmd="qsub -hard -e . -o . -l h_cpu=02:00:00 $script"
       
   fi 

   echo $cmd
   eval $cmd

  
}



batch-lookup(){

   local qwn=$1
   shift
   local path=$1
   local stamp=$2
   
   local jobs=$HOME
   local data=$OUTPUT_BASE
   
   local branch=$path/$stamp
   local databranch=$data/$branch
   local jobsbranch=$jobs/$branch
  
   eval val=\$$qwn
   echo $val
}

batch-prepfold(){
   
   local  path=$1
   local stamp=$2
   
   local iwd=$(pwd)
   
   ## parallel heirarchies, for job metadata and the data 
   
   local jobs=$(batch-lookup jobs $*) 
   local data=$(batch-lookup data $*)
   local branch=$(batch-lookup branch $*)
   local jobsbranch=$(batch-lookup jobsbranch $*) 
   local databranch=$(batch-lookup databranch $*)
   
   echo ==== batch-prepfold jobs:$jobs data:$data branch:$branch jobsbranch:$jobsbranch databranch:$databranch
   
   cd $data &&  mkdir -p $branch  && cd $path && rm -f last && ln -s $stamp last && cd $stamp 
   cd $jobs &&  mkdir -p $branch  && cd $path && rm -f last && ln -s $stamp last && cd $stamp 
   
   ## cross linking for convenience
   cd $databranch && ln -s $jobsbranch jobs
   cd $jobsbranch && ln -s $databranch data

   cd $jobs && rm -f last_jobs && ln -s $jobsbranch last_jobs 
   cd $jobs && rm -f last_data && ln -s $databranch last_data

   cd $iwd
}







batch-script(){

#
# convert arguments : 
# 
#      1) relative-path-classification
#    2..)  command line (such as  a function invokation)
#
# into a script for batch submission
# with xml logging 
#
#   -l means, behave like a login script
cat << EOC
#!/bin/bash 
## because condor forgets its home in job submission
export HOME=$HOME
echo ============= start ======  HOME \$HOME
env 
. $HOME/env/env.bash
echo ============ after env setup ======
env
echo =========== functions ========
declare -f 

#batch-logged-task $*

EOC

}







hello(){
  echo ========== hello $*
  echo ========== pwd
  pwd
  echo ========== env
  env 
}


batch-timestamp(){
	perl -MPOSIX -e  "print strftime( '%Y%m%d-%H%M%S' , localtime(time()) );" 
}

batch-fmtime(){
	perl -MPOSIX -e  "print strftime( '%Y%m%d-%H%M%S' , localtime($1) );" 
}


batch-logged-task(){

   ## runs the arguments passed sandwiched between xml logging information

   local path=$1    ## relative path describing  the task 
   shift

   printf "<task path=\"%s\"  >\n" $path 
   printf "<pwd>%s</pwd>\n" $(pwd)
   printf "<dirname>%s</dirname>\n" $(dirname $(pwd))
   printf "<basename>%s</basename>\n" $(basename $(pwd))
   
   printf "<args>\n" 
   for arg in "$@"
   do
	  printf "<arg>%s</arg>\n" $arg   
   done
   printf "</args>\n" 
   
   local start=$(date +'%s')
   printf "<start stamp=\"%d\" time=\"%s\" />\n" $start $(batch-fmtime $start)  
   printf "<body>\n" 
   
   eval "$@"
   
   printf "</body>\n" 
   local finish=$(date +'%s')
   printf "<finish stamp=\"%d\" time=\"%s\" />\n" $finish $(batch-fmtime $finish)  
   printf "</task>\n" 
}






