# === func-gen- : gallery3/gallery3 fgp gallery3/gallery3.bash fgn gallery3
gallery3-src(){      echo gallery3/gallery3.bash ; }
gallery3-source(){   echo ${BASH_SOURCE:-$(env-home)/$(gallery3-src)} ; }
gallery3-vi(){       vi $(gallery3-source) ; }
gallery3-env(){      elocal- ; }
gallery3-usage(){
  cat << EOU
     gallery3-src : $(gallery3-src)


     http://kohanaphp.com
     http://github.com/gallery/gallery3/tree/master

    Needs :
       PHP 5.2.3 
     so its a nogo from distros ...
         C ... php 4.3.9 
         N ... php 5.1.6


EOU
}


gallery3-dir(){ echo $(local-base)/env/gallery3/gallery3 ; }
gallery3-cd(){  cd $(gallery3-dir) ; }
gallery3-url(){ echo git://github.com/gallery/gallery3.git ; }
gallery3-get(){
  local dir=$(dirname $(gallery3-dir)) && mkdir -p $dir
  cd $dir
  git clone  $(gallery3-url) 
}


