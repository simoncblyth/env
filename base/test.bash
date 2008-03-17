


test-cp(){

  #
  #   without arguments :
  #         creates a file of 100Mb and copies it, timing the action
  #
  #   with argument (note no tailing slash):
  #          test-cp /tmp/tt
  #          test-cp /Volumes/Hello 
  #
  #              copies the test file to the provided directory path (that is assumed to exist) 
  #

  local msg="=== $FUNCNAME :"
  echo $msg 
  python $HOME/$BASE_BASE/test-cp.py  $* 
   
}