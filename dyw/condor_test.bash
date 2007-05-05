
condor-test-x(){ scp $HOME/$DYW_BASE/condor_test.bash ${1:-$TARGET_TAG}:$DYW_BASE ; }
condor-test-i(){ .   $HOME/$DYW_BASE/condor_test.bash ; }

x-cmd-test(){

  ## OK	
  x-cmd G1 pwd
  x-cmd G1 env
  x-cmd G1 logged-task g4dyb 

  ## CMT complaining ??? fixed by "set -- "
  x-cmd G1 logged-task g4dyb test_aberdeen two three four five six seven

}

x-cmd(){
 X=${1:-$TARGET_TAG}
 shift
 ssh $X "bash -lc '$@'"
}





condor-test-args(){
  f=$1
  shift	
  echo first argument is $f , the rest are :
  for arg in $@
  do
	 echo $arg   
  done
 } 



condor-test-vanilla(){

   tmp=/tmp/condor-test-vanilla-$$
   mkdir $tmp && cd $tmp

cat << EOF > hello.c
#include <stdio.h>
int main(void){
	extern char **environ;
	printf("hello, Condor\n");
    char  **envp; for (envp = environ; envp && *envp; envp++) printf("%s\n", *envp); 	
	return 0;
}
EOF

   gcc -o hello hello.c
   ./hello

   cat <<EOS > submit.hello
Executable     = hello
Universe       = vanilla 
Output         = hello.out
Log            = hello.log 
Queue 
EOS
   which condor_submit
   condor_submit submit.hello

   # echo sleeping for some seconds
   # sleep 5
   # cid=$(condor_q -f "%s" ClusterId)
   # condor_rm $cid
}



hello(){
  
  iwd=$(pwd) 
  cd $USER_BASE/condor/test
  
  echo hello $@
  pwd

  echo hello > hello.created
  uname -a
  #ls -alst world.out
  #cat world.out

  cd $iwd
}

world(){
  iwd=$(pwd) 
  cd $USER_BASE/condor/test

  echo world $@
  echo world > world.created
  pwd
  uname -a 
  ls -alst hello.out
  cat hello.out

  cd $iwd
}

condor-test-dag(){

   local func=$1
   local post=$2
   
   condor-use-func $func > $func.sub
   condor-use-func $post > $post.sub

   condor-use-dag  $func $post > $func.dag 
}







condor-test-redirection(){

  echo hello > hello

  eval " ( echo hello && ls -1 asjhdxa ) 1> out 2> err "

}


condor-test-submit(){
  condor-use-submit condor-test-redirection
}
	


condor-test-standard(){

# http://www.cs.wisc.edu/condor/quick-start.html

   cd $CONDOR_BASE 
   test -d condor || $SUDO mkdir condor
   
   cd condor
   mkdir -p test-standard 
   cd test-standard

   cat << EOF > hello.c
#include <stdio.h>
int main(void){
	extern char **environ;
	printf("hello, Condor\n");
    char  **envp; for (envp = environ; envp && *envp; envp++) printf("%s\n", *envp); 	
	return 0;
}
EOF

   gcc -o hello hello.c
   ./hello

   which condor_compile
   condor_compile gcc hello.c -o hello

   ./hello

  
   cat <<EOS > submit.hello
########################
# Submit description file for hello program
########################
Executable     = hello
Universe       = standard
Output         = hello.out
Log            = hello.log 
Queue 
EOS

   which condor_submit
   condor_submit submit.hello

}


condor-cmds(){
   ls $(dirname $(which condor))
}







