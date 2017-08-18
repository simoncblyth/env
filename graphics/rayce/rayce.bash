# === func-gen- : graphics/rayce/rayce fgp graphics/rayce/rayce.bash fgn rayce fgh graphics/rayce
rayce-src(){      echo graphics/rayce/rayce.bash ; }
rayce-source(){   echo ${BASH_SOURCE:-$(env-home)/$(rayce-src)} ; }
rayce-vi(){       vi $(rayce-source) ; }
rayce-env(){      elocal- ; }
rayce-usage(){ cat << EOU



* https://mathoverflow.net/questions/43552/anuloid-torus-x-line-intersection
* https://hanwen.home.xs4all.nl/public/software/README.html
* https://hanwen.home.xs4all.nl/public/software/rayce-3.0.tar.gz


EOU
}
rayce-dir(){ echo $(local-base)/env/graphics/rayce-3.0 ; }
rayce-cd(){  cd $(rayce-dir); }
rayce-mate(){ mate $(rayce-dir) ; }
rayce-get(){
   local dir=$(dirname $(rayce-dir)) &&  mkdir -p $dir && cd $dir

   local url=https://hanwen.home.xs4all.nl/public/software/rayce-3.0.tar.gz
   local dst=$(basename $url)
   [ ! -f "$dst" ] && curl -L -O $url

   local nam=${dst/.tar.gz}
   [ ! -d "$nam" ] && tar zxvf $dst

 


}
