
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


condor-use-prepfold(){

   local path=$1
   local stamp=$2
   
   iwd=$(pwd)
   
   ## parallel heirarchies, HOME is local and USER_BASE is network mounted
   
   cd $USER_BASE &&  mkdir -p $path/$stamp  && cd $path && rm -f last && ln -s $stamp last && cd $stamp
   cd $HOME      &&  mkdir -p $path/$stamp  && cd $path && rm -f last && ln -s $stamp last && cd $stamp 
   
   cd $USER_BASE/$path/$stamp && ln -s $HOME/$path/$stamp sub
   cd $HOME/$path/$stamp      && ln -s $USER_BASE/$path/$stamp out 

   cd $HOME && rm -f last_sub && ln -s $HOME/$path/$stamp last_sub 
   cd $HOME && rm -f last_out && ln -s $USER_BASE/$path/$stamp last_out 

   cd $iwd
}

condor-use-submit(){

   local  path=$1
   local  func=$2
   local stamp=$(condor-use-timestamp)

   shift

   [ "X$path" == "X" ] && echo must provide a relative path for classification && return 
   [ "X$func" == "X" ] && echo must provide a function to call                 && return 
 
   condor-use-prepfold $path $stamp
   idir=$USER_BASE/$path/$stamp
   sdir=$HOME/$path/$stamp
   cd $sdir

   condor-use-func $idir "$@" > $func.sub

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



condor-use-script(){

   local idir=$1
   [ "X$idir" == "X" ] && echo condor-use-script must provide a idir initial directory for the job && return 
   shift
#
#  this dyb specific stuff should not be here ...
#
#   -l means, behave like a login script

cat << EOC
#!/bin/bash 
export HOME=$HOME

iwd=\$(pwd)
[ -r ~/env/env.bash ] && . ~/env/env.bash
cd \$iwd

cd $idir 

condor-use-logged-task $@


EOC

}



condor-use-logged-task(){

   ## runs the arguments passed sandwiched between xml logging information

   func=${1:-condor-use-logged-task}

   printf "<task func=\"%s\"  >\n" $func 
   printf "<pwd>%s</pwd>\n" $(pwd)
   printf "<dirname>%s</dirname>\n" $(dirname $(pwd))
   printf "<basename>%s</basename>\n" $(basename $(pwd))
   
   printf "<args>\n" 
   for arg in "$@"
   do
	  printf "<arg>%s</arg>\n" $arg   
   done
   printf "</args>\n" 
   
   start=$(date +'%s')
   printf "<start stamp=\"%d\" time=\"%s\" />\n" $start $(condor-use-fmtime $start)  
   printf "<body>\n" 
   
   eval "$@"
   
   printf "</body>\n" 
   finish=$(date +'%s')
   printf "<finish stamp=\"%d\" time=\"%s\" />\n" $finish $(condor-use-fmtime $finish)  
   printf "</task>\n" 
}



condor-use-timestamp(){
	perl -MPOSIX -e  "print strftime( '%Y%m%d-%H%M%S' , localtime(time()) );" 
}

condor-use-fmtime(){
	perl -MPOSIX -e  "print strftime( '%Y%m%d-%H%M%S' , localtime($1) );" 
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
  local func=$2 

  [ "X$idir" == "X" ] && echo condor-use-subfile must provide a idir initial directory for the job && return 
  [ "X$func" == "X" ] && echo condor-use-subfile must provide a function to call                   && return 

  
cat << EOS
##########################################
# Condor Submit description func:$func 
# args:$@  
##########################################

## attempt non shared filesystem 
should_transfer_files = YES
when_to_transfer_output = ON_EXIT_OR_EVICT

Executable     = /bin/bash
Arguments      = $func.sh
Universe       = vanilla
initialdir     = $idir
Output         = $func.out
Log            = $func.log
Log_Xml        = True
Error          = $func.err
Queue 
EOS
}


