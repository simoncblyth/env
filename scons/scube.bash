# === func-gen- : scons/scube fgp scons/scube.bash fgn scube fgh scons
scube-src(){      echo scons/scube.bash ; }
scube-source(){   echo ${BASH_SOURCE:-$(env-home)/$(scube-src)} ; }
scube-vi(){       vi $(scube-source) ; }
scube-env(){      elocal- ; }
scube-usage(){
  cat << EOU
     scube-src : $(scube-src)
     scube-dir : $(scube-dir)


EOU
}

scube-url(){ echo http://scube.googlecode.com/svn/trunk/ ; }

scube-dir(){ echo $(local-base)/env/scons/scube ; }
scube-cd(){  cd $(scube-dir); }
scube-mate(){ mate $(scube-dir) ; }
scube-get(){
   local dir=$(dirname $(scube-dir)) &&  mkdir -p $dir && cd $dir

   [ ! -d "scube" ] && svn co $(scube-url) scube
}
