# === func-gen- : nodejs/nodeamqp fgp nodejs/nodeamqp.bash fgn nodeamqp fgh nodejs
nodeamqp-src(){      echo nodejs/nodeamqp.bash ; }
nodeamqp-source(){   echo ${BASH_SOURCE:-$(env-home)/$(nodeamqp-src)} ; }
nodeamqp-vi(){       vi $(nodeamqp-source) ; }
nodeamqp-env(){      elocal- ; }
nodeamqp-usage(){
  cat << EOU
     nodeamqp-src : $(nodeamqp-src)
     nodeamqp-dir : $(nodeamqp-dir)

     Using squaremo fork for use with rabbit.js


 == node-amqp issues == 

   TypeError fixed by adding parent. as suggested by 
       http://github.com/ry/node-amqp/issues#issue/7 

/data1/env/local/env/nodejs/node-amqp/amqp.js:263
  var s = buffer.utf8Slice(buffer.read, buffer.read+length);
                 ^
/data1/env/local/env/nodejs/node-amqp/amqp.js:511
  b.utf8Write(string, b.used); // error here
    ^

== trying node-amqp example from synopsis http://github.com/squaremo/node-amqp ==


   node synopsis.js 

   Unhandled channel error:





EOU
}
nodeamqp-dir(){ echo $(local-base)/env/nodejs/nodejs-nodeamqp ; }
nodeamqp-cd(){  cd $(nodeamqp-dir); }
nodeamqp-mate(){ mate $(nodeamqp-dir) ; }
nodeamqp-get(){
   local dir=$(dirname $(nodeamqp-dir)) &&  mkdir -p $dir && cd $dir

    git clone http://github.com/squaremo/node-amqp.git

}


nodeamqp-cfg-export(){
  private-
  local vars="AMQP_SERVER AMQP_PORT AMQP_USER AMQP_PASSWORD AMQP_VHOST"
  local var 
  for var in $vars ; do 
    local exp="export $var=$(private-val $var)"
    echo $exp
    eval $exp
  done
}


