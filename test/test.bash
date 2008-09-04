
test-usage(){

 cat << EOU

    test-deploy 
        copy the .py from here into dybtest in nuwa wc


EOU

}


test-env(){
  elocal-
}

test-deploy(){

  local msg="# === $FUNCNAME :"
  local iwd=$PWD
  [ -z $NUWA_HOME ] && echo $msg ABORT no NUWA_HOME && return 1 
  cd $ENV_HOME/test
  
  local rel=dybgaudi/DybTest/python/dybtest
  
  echo $msg pipe this to sh to do the copy 
  local pys="match.py run.py runner.py count.py"
  for py in $pys ; do
    local cmd="cp $ENV_HOME/test/$py $NUWA_HOME/$rel/ "
    echo $cmd  
  done


  cd $iwd
}
