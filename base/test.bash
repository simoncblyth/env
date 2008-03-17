


test-cp(){

  local msg="=== $FUNCNAME :"
  echo $msg 
  python $HOME/$BASE_BASE/test-cp.py  $* 
   
}