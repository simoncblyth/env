# === func-gen- : nodejs/rabbitjs fgp nodejs/rabbitjs.bash fgn rabbitjs fgh nodejs
rabbitjs-src(){      echo nodejs/rabbitjs.bash ; }
rabbitjs-source(){   echo ${BASH_SOURCE:-$(env-home)/$(rabbitjs-src)} ; }
rabbitjs-vi(){       vi $(rabbitjs-source) ; }
rabbitjs-env(){      elocal- ; }
rabbitjs-usage(){
  cat << EOU
     rabbitjs-src : $(rabbitjs-src)
     rabbitjs-dir : $(rabbitjs-dir)


     http://github.com/squaremo/rabbit.js


    Using squaremo node-amqp fork , see nodeamqp-




EOU
}
rabbitjs-dir(){ echo $(local-base)/env/nodejs/rabbit.js ; }
rabbitjs-cd(){  cd $(rabbitjs-dir); }
rabbitjs-mate(){ mate $(rabbitjs-dir) ; }
rabbitjs-get(){
   local dir=$(dirname $(rabbitjs-dir)) &&  mkdir -p $dir && cd $dir

   git clone http://github.com/squaremo/rabbit.js.git
  # git clone http://github.com/LearnBoost/Socket.IO-node.git --recursive
}
