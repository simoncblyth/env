
condor-use-x(){ scp $HOME/$DYW_BASE/condor_use.bash ${1:-$TARGET_TAG}:$DYW_BASE ; }
condor-use-i(){ .   $HOME/$DYW_BASE/condor_use.bash ; }

#
#
#    condor-use-submit
#       condor-use-script
#          condor-use-logged-task        runs the arguments passed sandwiched between xml logging information
#             condor-use-timestamp
#             condor-use-fmtime
#       condor-use-subfile func idir
#
#
#          usage:
#               condor-use-submit func arguments
#
#                  where func is a bash function 
#
#                  creates a timestamped folder within the current directory in which 
#                  the condor subfile func.sub is created where func is the first
#                  argument passed ... other job outputs are called
#                        func.{out,err,log}
#
#
#   issues:
#
#      1) better bookeeping ... maybe copy the macro into the run folder at
#         run start, and thus the .root should be created there ???
#         ensure all jobs outputs end up in the folder 
#
#
#      2) xml submission report , with references to the output file paths
#         ... that can be curl -T into exist , whence can stylesheet
#         a table of submissions
#
#      3) move to depending on far fewer files to define the environment
#       
#      4) split the .bash_* into _{use,build} sections ???
#
#      5) some funny characters in the logs prevents valid xml
#
#

#  Current status "ST" of the job. 
#    U = unexpanded (never been run), 
#    I = idle (waiting for a machine to execute on), 
#    R = running, 
#    H =  on hold, 
#    C = completed, and 
#    X = removed.
#
#   use the "ID" given by condor_q to remove jobs
#   condor_rm ID
#


condor-use-lookup(){

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


condor-use-prepfold(){
   
   local  path=$1
   local stamp=$2
   
   iwd=$(pwd)
   
   ## parallel heirarchies, HOME is local and USER_BASE is network mounted
   
   local jobs=$(condor-use-lookup jobs $*) 
   local data=$(condor-use-lookup data $*)
   local branch=$(condor-use-lookup branch $*)
   local jobsbranch=$(condor-use-lookup jobsbranch $*) 
   local databranch=$(condor-use-lookup databranch $*)
   
   
   echo ==== condor-use-prepfold jobs:$jobs data:$data branch:$branch jobsbranch:$jobsbranch databranch:$databranch
   
   cd $data &&  mkdir -p $branch  && cd $path && rm -f last && ln -s $stamp last && cd $stamp
   cd $jobs &&  mkdir -p $branch  && cd $path && rm -f last && ln -s $stamp last && cd $stamp
   
   ## cross linking for convenience
   cd $databranch && ln -s $jobsbranch jobs
   cd $jobsbranch && ln -s $databranch data

   cd $jobs && rm -f last_jobs && ln -s $jobsbranch last_jobs 
   cd $jobs && rm -f last_data && ln -s $databranch last_data

   cd $iwd
}

condor-use-test(){

  ## invoke with condor-use-submit jobs/test condor-use-test
  echo ====== condor-use-test ========= test function
  pwd
  ls -alst 

}

condor-use-submit(){

   ## this is the entry point for instrumented condor job submission 
   
   local  path=$1    ## relative path that acts as a classification string for the job
   local  func=$2    ## bash function to be invoked
   
   local stamp=$(condor-use-timestamp)
   echo =========== condor-use-submit path:$path func:$func stamp:$stamp 

   shift

   [ "X$path" == "X" ] && echo must provide a relative path for classification && return 
   [ "X$func" == "X" ] && echo must provide a function to call                 && return 
 
   condor-use-prepfold $path $stamp
   
   local jobsbranch=$(condor-use-lookup jobsbranch $path $stamp) 
   local databranch=$(condor-use-lookup databranch $path $stamp)
      
   echo ============ condor-use-submit jobsbranch:$jobsbranch databranch:$databranch    
      
   ## getting 
   ##  Error from starter on albert4.hepgrid: Failed to open standard output file '/disk/d4/blyth/jobs/test/20070613-160016/condor-use-test.out': Permission denied (errno 13)<   
   ##cd $jobsbranch
   cd $databranch
   condor-use-func $databranch "$@" > $func.sub

   echo ============ finally the real submission to condor  

   condor_submit  $func.sub
}

condor-use-submit-pair(){

#
#  usage example :
#
#      condor-use-submit-pair jobs/test/helloworld world hello simon and more args
#
#   the "hello" function runs first with arguments : simon and more args
#   after it is completed the "world" function runs
#
#   condor-dag cannot be submitted from network mounted volume ...
#   but i wanted for the 2nd job to access the output from the first 
# 
#
#
#
   local path=$1
   local post=$2
   local func=$3
   local stamp=$(condor-use-timestamp)

   [ "X$path" == "X" ] && echo must provide a relative path for classification && return 
   [ "X$post" == "X" ] && echo must provide a post-function to call            && return 
   [ "X$func" == "X" ] && echo must provide a function to call                 && return 

   shift
   shift

   condor-use-prepfold $path $stamp
   idir=$USER_BASE/$path/$stamp
   sdir=$HOME/$path/$stamp
   cd $sdir
   
   condor-use-func $idir "$@"  > $func.sub
   condor-use-func $idir $post > $post.sub
   condor-use-pair       $func $post > $func.dag 

   condor_submit_dag $func.dag
}


condor-use-func(){

   local idir=$1
   local func=$2
   
   [ "X$idir" == "X" ] && echo condor-use-func must provide a idir initial directory for the job && return 
   [ "X$func" == "X" ] && echo condor-use-func must provide a function to call                   && return 
   
   shift

   ## handle the arguments in a .sh rather than the .sub file as its a pain to quote the parameters correctly
   condor-use-script  $idir "$@"  > $idir/$func.sh
   condor-use-subfile $idir $func  
}


condor-use-pair(){

cat << EOD  
JOB A $1.sub
JOB B $2.sub
PARENT A CHILD B
EOD

}















condor-use-subfile(){
#
#   using the new form of condor arguments ...
#   with quotes around all arguments and literal quotes escaped by repetition
#      http://www.cs.wisc.edu/condor/manual/v6.7/condor_submit.html
#
#  failed to get quoting to work ..
# Arguments      = -lc \"logged-task $@ \" 
#  so move to generating script to run	
#  
  local idir=$1
  local name=$2 
  local script=$3

  [ "X$idir" == "X" ] && echo condor-use-subfile must provide a idir initial directory for the job && return 1
  [ "X$name" == "X" ] && echo condor-use-subfile must provide a name                               && return 1
  [ "X$script" == "X" ] && echo condor-use-subfile must provide a script                           && return 1           



  
cat << EOS
##########################################
# Condor Submit description func:$func 
# args:$@  
##########################################

## attempt non shared filesystem 
#should_transfer_files = YES
#when_to_transfer_output = ON_EXIT_OR_EVICT

Executable     = /bin/bash
Arguments      = $script
Universe       = vanilla
initialdir     = $idir
Output         = $name.out
Log            = $name.log
Log_Xml        = True
Error          = $name.err
Queue 
EOS
}


