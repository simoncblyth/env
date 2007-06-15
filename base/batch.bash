

batch-script-write(){

   local path=$1
   local func=$2
   
   batch-script > $func.sh    
}



hello(){
  echo ========== hello $*
  echo ========== pwd
  pwd
  echo ========== env
  env 
}



batch-script(){

   local path=$1
   local func=$2
   
   [ "X$path" == "X" ] && echo need relative path that describes the task   && return 1
   [ "X$func" == "X" ] && echo need func to perform                         && return 1 
   
   shift
   shift

#   -l means, behave like a login script
cat << EOC
#!/bin/bash -l
$func $*

EOC

}

