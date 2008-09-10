
dtracebuild-usage(){

   cat << EOU

        dtracebuild-name : $(dtracebuild-name)
        dtracebuild-url  : $(dtracebuild-url)
        dtracebuild-dir  : $(dtracebuild-dir) 

        dtracebuild-get/configure/install


EOU

}

dtracebuild-env(){
  echo -n
}

dtracebuild-name(){ echo dtrace-20080803 ; }
dtracebuild-ball(){ echo $(dtracebuild-name).tar.bz2 ; }
dtracebuild-url(){  echo ftp://crisp.dynalias.com/pub/release/website/dtrace/$(dtracebuild-ball) ; }
dtracebuild-dir(){  echo $SYSTEM_BASE/dtrace/build/$(dtracebuild-name) ; }
dtracebuild-cd(){   cd $(dtracebuild-dir) ; }


dtracebuild-get(){

    local dir=$(dtracebuild-dir)
    local ball=$(dtracebuild-ball)
    local url=$(dtracebuild-url)
    local nam=$(dtracebuild-name)

    mkdir -p $(dirname $dir)
    cd $(dirname $(dirname $dir))

    [ ! -f $ball ] && curl -O $url
    [ ! -d build/$nam ] && ( cd build ; bunzip2 $ball ) 
}






