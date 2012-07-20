# === func-gen- : nodejs/njs fgp nodejs/njs.bash fgn njs fgh nodejs
njs-src(){      echo nodejs/njs.bash ; }
njs-source(){   echo ${BASH_SOURCE:-$(env-home)/$(njs-src)} ; }
njs-vi(){       vi $(njs-source) ; }
njs-env(){      elocal- ; }
njs-usage(){ cat << EOU

Node.js 
========

Earlier experiments in ``nodejs-`` 

http://nodejs.org/docs/v0.6.19/api/







EOU
}
njs-dir(){ echo $(local-base)/env/nodejs/$(njs-name) ; }
njs-cd(){  cd $(njs-dir); }
njs-vers(){ echo v0.6.19 ; }
njs-name(){ echo node-$(njs-vers) ; }
njs-url(){  echo http://nodejs.org/dist/$(njs-vers)/$(njs-name).tar.gz ;}
njs-mate(){ mate $(njs-dir) ; }
njs-get(){
   local dir=$(dirname $(njs-dir)) &&  mkdir -p $dir && cd $dir
   local url=$(njs-url)
   local tgz=$(basename $url)
   local nam=$(njs-name)
  
   [ ! -f "$tgz" ] && curl -L -O $url
   [ ! -d "$nam" ] && tar zxvf $nam.tar.gz

}
