# === func-gen- : tools/mermaid/mermaid fgp tools/mermaid/mermaid.bash fgn mermaid fgh tools/mermaid src base/func.bash
mermaid-source(){   echo ${BASH_SOURCE} ; }
mermaid-edir(){ echo $(dirname $(mermaid-source)) ; }
mermaid-ecd(){  cd $(mermaid-edir); }
mermaid-dir(){  echo $LOCAL_BASE/env/tools/mermaid ; }
mermaid-cd(){   cd $(mermaid-dir); }
mermaid-vi(){   vi $(mermaid-source) ; }
mermaid-env(){  elocal- ; }
mermaid-usage(){ cat << EOU



* https://yarnpkg.com/lang/en/docs/usage/



* https://stackoverflow.com/questions/52057634/failed-to-download-chromium-r579032

* https://github.com/GoogleChrome/puppeteer/issues/1597

yarn config set puppeteer_download_host=https://npm.taobao.org/mirrors

* seemed to work to avoid Chromium blockage   


EOU
}
mermaid-get(){
   local dir=$(mermaid-dir) &&  mkdir -p $dir && cd $dir

   [ ! -d node_modules/mermaid ] && yarn add mermaid
   [ ! -d node_modules/mermaid.cli ] && yarn add mermaid.cli
   [ ! -d node_modules/mermaid-live-editor ] && yarn add mermaid-live-editor 

}

mermaid-mmdc(){
    mermaid-cd
    ./node_modules/.bin/mmdc -h

}



mermaid-example-(){ cat << EOE

 graph TD;
        A-->B;
        A-->C;
        B-->D;
        C-->D;

EOE
}

mermaid-example(){

  local tmp=/tmp/env/mermaid 
  mkdir -p $tmp

  local path=$tmp/$FUNCNAME.mmd

  $FUNCNAME- > $path
 
  mermaid-cd

  ./node_modules/.bin/mmdc -i $path -o ${path/.mmd/.png}   


}



