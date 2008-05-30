
bitten-usage(){

cat << EOU





EOU
}


bitten-env(){
  elocal-
}


bitten-get(){

  local iwd=$PWD
  
  cd $HOME
  [ ! -d bitten ] && svn co http://svn.edgewall.org/repos/bitten/trunk bitten

  cd bitten
  svn info
  svn up



  cd $iwd

}



