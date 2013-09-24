# === func-gen- : npy/mysql_numpy fgp npy/mysql_numpy.bash fgn mysql_numpy fgh npy
mysql_numpy-src(){      echo npy/mysql_numpy.bash ; }
mysql_numpy-source(){   echo ${BASH_SOURCE:-$(env-home)/$(mysql_numpy-src)} ; }
mysql_numpy-vi(){       vi $(mysql_numpy-source) ; }
mysql_numpy-env(){      elocal- ; }
mysql_numpy-usage(){ cat << EOU





EOU
}
mysql_numpy-dir(){ echo $(local-base)/env/npy/mysql_numpy ; }
mysql_numpy-cd(){  cd $(mysql_numpy-dir); }
mysql_numpy-mate(){ mate $(mysql_numpy-dir) ; }
mysql_numpy-get(){
   local dir=$(dirname $(mysql_numpy-dir)) &&  mkdir -p $dir && cd $dir

   git clone git://github.com/scb-/mysql_numpy.git

}
