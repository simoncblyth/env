# === func-gen- : network/asio/asio fgp network/asio/asio.bash fgn asio fgh network/asio
asio-src(){      echo network/asio/asio.bash ; }
asio-source(){   echo ${BASH_SOURCE:-$(env-home)/$(asio-src)} ; }
asio-vi(){       vi $(asio-source) ; }
asio-env(){      elocal- ; }
asio-usage(){ cat << EOU

Non-boost ASIO
===============

* https://github.com/chriskohlhoff/asio

* See also basio-

asio version 1.10.6
Released Tuesday, 24 March 2015.

* https://rogiel.com/blog/getting-started-with-asio-cpp-creating-tcp-server


EOU
}
asio-dir(){ echo $(local-base)/env/network/$(asio-name); }
asio-fold(){ echo $(dirname $(asio-dir)) ; }
asio-fcd(){ 
    local fold=$(asio-fold)
    mkdir -p $fold
    cd $fold
 }
asio-cd(){  cd $(asio-dir); }
asio-mate(){ mate $(asio-dir) ; }
asio-name(){ echo asio-1.10.6 ; }
asio-doc(){ open $(asio-dir)/doc/index.html ; }

asio-get(){
   local dir=$(dirname $(asio-dir)) &&  mkdir -p $dir && cd $dir

   local url="http://downloads.sourceforge.net/project/asio/asio/1.10.6%20%28Stable%29/asio-1.10.6.zip"


   local name=$(asio-name)

   [ ! -d "$name" ] &&  cat << EOM  

ARGH funny chars in URL, use GUI

    asio-fcd
    # download using Safari
    mv ~/Downloads/asio-1.10.6 .

EOM


}
