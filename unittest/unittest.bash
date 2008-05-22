
unittest-usage(){

  cat << EOU




EOU



}


unittest-env(){

  elocal-
}




dip-get(){



  local tmp=/tmp/$FUNCNAME && mkdir -p $tmp
  local iwd=$PWD
  cd $tmp

  local nam=diveintopython-5.4
  local zip=diveintopython-examples-5.4.zip
  local url=http://diveintopython.org/download/$zip


  [ ! -f $zip ] && curl -O $url
  [ ! -d $nam ] && unzip $zip

  cd $nam/py
  
  python romantest.py 
#
# .............
# ----------------------------------------------------------------------
# Ran 13 tests in 0.601s
#
# OK
#


  #cd $iwd
}





