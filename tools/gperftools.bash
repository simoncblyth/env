# === func-gen- : tools/gperftools fgp tools/gperftools.bash fgn gperftools fgh tools
gperftools-src(){      echo tools/gperftools.bash ; }
gperftools-source(){   echo ${BASH_SOURCE:-$(env-home)/$(gperftools-src)} ; }
gperftools-vi(){       vi $(gperftools-source) ; }
gperftools-env(){      elocal- ; }
gperftools-usage(){ cat << EOU

GPERFTOOLS
============



EOU
}
gperftools-dir(){ echo $(local-base)/env/tools/$(gperftools-name) ; }
gperftools-cd(){  cd $(gperftools-dir); }
gperftools-mate(){ mate $(gperftools-dir) ; }
gperftools-name(){ echo gperftools-2.0 ;}
gperftools-get(){
   local dir=$(dirname $(gperftools-dir)) &&  mkdir -p $dir && cd $dir
   local nam=$(gperftools-name)
   local tgz=$nam.tar.gz
   local url=http://gperftools.googlecode.com/files/$tgz

   [ ! -f "$tgz" ] && curl -L -O "$url"
   [ ! -d "$nam" ] && tar zxvf $tgz
}
