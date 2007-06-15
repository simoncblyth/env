

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


batch-timestamp(){
	perl -MPOSIX -e  "print strftime( '%Y%m%d-%H%M%S' , localtime(time()) );" 
}

batch-fmtime(){
	perl -MPOSIX -e  "print strftime( '%Y%m%d-%H%M%S' , localtime($1) );" 
}


batch-logged-task(){

   ## runs the arguments passed sandwiched between xml logging information

   local name=$1

   printf "<task name=\"%s\"  >\n" $name 
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

