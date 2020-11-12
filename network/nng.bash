# === func-gen- : network/nng fgp network/nng.bash fgn nng fgh network src base/func.bash
nng-source(){   echo ${BASH_SOURCE} ; }
nng-edir(){ echo $(dirname $(nng-source)) ; }
nng-ecd(){  cd $(nng-edir); }

nng-prefix(){  echo $LOCAL_BASE/env/network/nng ; }
nng-dir(){     echo $(nng-prefix)_src ; }
nng-sdir(){    echo $(nng-prefix)_src ; }
nng-bdir(){    echo $(nng-prefix)_build ; }

nng-cd(){   cd $(nng-dir); }
nng-scd(){  cd $(nng-sdir); }
nng-bcd(){  cd $(nng-bdir); }

nng-vi(){   vi $(nng-source) ; }
nng-env(){  elocal- ; }
nng-usage(){ cat << EOU


* https://github.com/nanomsg/nng
* https://nng.nanomsg.org/


* https://github.com/search?q=nng+nanomsg&type=


* https://github.com/robiwano/siesta

  Siesta is a minimalistic HTTP, REST and Websocket framework for C++, written in pure-C++11, based upon NNG 

* https://github.com/ilyaevseev/nngpy

   Simple Python wrapper for nng (nanomsg next generation) library.

* https://github.com/graphiclife/nng-swift

* https://staysail.github.io/nng_presentation/nng_presentation.html



EOU
}

nng-url(){ echo https://github.com/nanomsg/nng ; }

nng-get(){
   local dir=$(dirname $(nng-sdir)) &&  mkdir -p $dir && cd $dir
   local name=$(basename $(nng-sdir))
   [ ! -d $name ] && git clone $(nng-url) $name 
}



nng-cmake()
{
    local sdir=$(nng-sdir)
    local bdir=$(nng-bdir)
    mkdir -p $bdir
    nng-bcd 
    cmake $sdir -G Ninja  -DCMAKE_INSTALL_PREFIX=$(nng-prefix)
}
nng-make()
{
    local msg="=== $FUNCNAME :"
    nng-bcd

    ninja
    [ $? -ne 0 ] && echo $msg ninja is required && exit 1 

    ninja test
    ninja install
}




nng-demo-info(){ cat << EOI


* https://nanomsg.org/gettingstarted/
* https://nanomsg.org/gettingstarted/nng/index.html

EOI
}



nng-demo()
{
   local rel=demo/async

   local sdir=$(nng-sdir)/$rel
   local bdir=/tmp/$USER/env/$FUNCNAME/$rel
   mkdir -p $bdir
   cd $bdir 
   CMAKE_PREFIX_PATH=$(nng-prefix) cmake $sdir 
   make 

   case $rel in 
       demo/async) cp $sdir/run.sh . ;;
   esac
}




