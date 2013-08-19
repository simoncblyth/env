# === func-gen- : tools/gperftools/gperftools fgp tools/gperftools/gperftools.bash fgn gperftools fgh tools/gperftools
gperftools-src(){      echo tools/gperftools/gperftools.bash ; }
gperftools-source(){   echo ${BASH_SOURCE:-$(env-home)/$(gperftools-src)} ; }
gperftools-vi(){       vi $(gperftools-source) ; }
gperftools-env(){      elocal- ; }
gperftools-usage(){ cat << EOU

GPERFTOOLS
============

* http://code.google.com/p/gperftools/
* http://code.google.com/p/gperftools/wiki/GooglePerformanceTools

INSTALLS
----------

belle7
~~~~~~~~
::

    gperftools-get
    gperftools-cd
    ./configure
    make

    [blyth@belle7 gperftools-2.1]$ sudo make install     ## installed into  /usr/local/lib/ including libprofiler.so 


CPU Profiling
-------------

* http://gperftools.googlecode.com/svn/trunk/doc/cpuprofile.html
* http://belle7.nuu.edu.tw/gperftools/pprof_remote_servers.html






EOU
}
gperftools-dir(){ echo $(local-base)/env/tools/gperftools/tools/$(gperftools-nam) ; }
gperftools-cd(){  cd $(gperftools-dir); }
gperftools-mate(){ mate $(gperftools-dir) ; }
gperftools-nam(){ echo gperftools-2.1 ; }
gperftools-get(){
   local dir=$(dirname $(gperftools-dir)) &&  mkdir -p $dir && cd $dir

   local nam=$(gperftools-nam)
   local tgz=$nam.tar.gz
   local url=http://gperftools.googlecode.com/files/$tgz
   [ ! -f "$tgz" ] && curl -L -O $url
   [ ! -d "$nam" ] && tar zxvf $tgz


}


gperftools-lib(){ echo /usr/local/lib/libprofiler.so ; }
