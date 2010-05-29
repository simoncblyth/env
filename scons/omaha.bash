# === func-gen- : scons/omaha fgp scons/omaha.bash fgn omaha fgh scons
omaha-src(){      echo scons/omaha.bash ; }
omaha-source(){   echo ${BASH_SOURCE:-$(env-home)/$(omaha-src)} ; }
omaha-vi(){       vi $(omaha-source) ; }
omaha-env(){      elocal- ; }
omaha-usage(){
  cat << EOU
     omaha-src : $(omaha-src)
     omaha-dir : $(omaha-dir)
   
   Omaha is windows only ... but it uses SCT/Scons for its build system
   making it an interesting real world usage example 

EOU
}
omaha-url(){ echo http://omaha.googlecode.com/svn/trunk/ ; }
omaha-dir(){ echo $(local-base)/env/scons/omaha ; }
omaha-cd(){  cd $(omaha-dir); }
omaha-mate(){ mate $(omaha-dir) ; }
omaha-get(){
   local dir=$(dirname $(omaha-dir)) &&  mkdir -p $dir && cd $dir
   [ ! -d omaha ] && svn co $(omaha-url) omaha
}

omaha-find(){
  local iwd=$PWD
  omaha-cd
  find . -name build.scons -exec grep -H $1 {} \;
  cd $iwd
}

