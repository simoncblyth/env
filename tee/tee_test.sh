#!/bin/bash -l 
usage(){ cat << EOU

https://serverfault.com/questions/201061/capturing-stderr-and-stdout-to-file-using-tee

EOU
}


command(){
printf "{%s}   This should MODE $MODE go to stderr.\n" "$(date)" 1>&2 
printf "[(%s)] This should MODE $MODE go to stdout.\n" "$(date)" 

}


launch()
{
   local tlog=/tmp/tee_test.log 

   expr="command  2>&1 | tee $tlog"

   eval $expr
 

   local  cmd="cat $tlog"
   echo $cmd
   eval $cmd

}

launch 


