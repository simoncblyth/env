
test-usage(){

 cat << EOU

    test-deploy 
        copy the .py from here into dybtest in nuwa wc


EOU

}

test-env(){ elocal- ; }
test-rel(){ echo dybgaudi/DybTest/python/dybtest ; }
test-pys(){ echo match.py run.py runner.py count.py command.py ; }

test-deploy(){

  local msg="# === $FUNCNAME :"
  local iwd=$PWD
  [ -z $NUWA_HOME ] && echo $msg ABORT no NUWA_HOME && return 1 
  cd $ENV_HOME/test
  
  echo $msg pipe this to sh to do the copy 
  for py in $(test-pys) ; do
    local cmd="cp $ENV_HOME/test/$py $NUWA_HOME/$(test-rel)/ "
    echo $cmd  
  done


  cd $iwd
}


test-diff(){

  local msg="# === $FUNCNAME :"
  local iwd=$PWD
  [ -z $NUWA_HOME ] && echo $msg ABORT no NUWA_HOME && return 1 

  cd $ENV_HOME/test
  for py in $(test-pys) ; do
     local cmd="diff $py $NUWA_HOME/$(test-rel)/$py "
     echo $cmd
     eval $cmd
  done    
  
  cd $iwd
}

test-count(){
   local iwd=$PWD
   cd $ENV_HOME/test
   
   local cmd="time python run.py \"python count.py 10\" 10"
   echo $cmd
   eval $cmd 

   cd $iwd
}

