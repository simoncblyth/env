gleq-src(){      echo externals/gleq.bash ; }
gleq-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(gleq-src)} ; }
gleq-vi(){       vi $(gleq-source) ; }
gleq-env(){      elocal- ; opticks- ; }
gleq-usage(){ cat << EOU



EOU
}
gleq-dir(){ echo $(opticks-prefix)/externals/gleq ; }
gleq-sdir(){ echo $(opticks-home)/graphics/gleq ; }
gleq-cd(){  cd $(gleq-dir); }
gleq-scd(){  cd $(gleq-sdir); }
gleq-edit(){ vi $(opticks-home)/cmake/Modules/FindGLEQ.cmake ; }


gleq-get(){
   local iwd=$PWD
   local dir=$(dirname $(gleq-dir)) &&  mkdir -p $dir && cd $dir
   [ ! -d gleq ] && git clone https://github.com/simoncblyth/gleq
   cd $iwd
}
gleq-hdr(){
   echo $(gleq-dir)/gleq.h
}

gleq--(){
   gleq-get
}

