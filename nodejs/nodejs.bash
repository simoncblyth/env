# === func-gen- : nodejs/nodejs fgp nodejs/nodejs.bash fgn nodejs fgh nodejs
nodejs-src(){      echo nodejs/nodejs.bash ; }
nodejs-source(){   echo ${BASH_SOURCE:-$(env-home)/$(nodejs-src)} ; }
nodejs-vi(){       vi $(nodejs-source) ; }
nodejs-env(){      elocal- ; }
nodejs-usage(){
  cat << EOU
     nodejs-src : $(nodejs-src)
     nodejs-dir : $(nodejs-dir)

     http://github.com/ry/node/tree/master 

EOU
}
nodejs-dir(){ echo $(local-base)/env/nodejs/node ; }
nodejs-cd(){  cd $(nodejs-dir); }
nodejs-mate(){ mate $(nodejs-dir) ; }

nodejs-url(){ echo git://github.com/ry/node.git ;  } 

nodejs-get(){
   local dir=$(dirname $(nodejs-dir)) &&  mkdir -p $dir && cd $dir

  git clone $(nodejs-url) 

}

nodejs-build(){
  nodejs-cd
  ./configure --prefix=$(local-base)/env

}


