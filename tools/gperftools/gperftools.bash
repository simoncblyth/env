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

G
~~~

No go on PPC::

    simon:gperftools-2.1 blyth$ sudo make
    Password:
    /bin/sh ./libtool  --tag=CXX   --mode=compile g++ -DHAVE_CONFIG_H -I. -I./src  -I./src   -DNO_TCMALLOC_SAMPLES -D_THREAD_SAFE  -DNDEBUG -Wall -Wwrite-strings -Woverloaded-virtual -Wno-sign-compare -fno-builtin-malloc -fno-builtin-free -fno-builtin-realloc -fno-builtin-calloc -fno-builtin-cfree -fno-builtin-memalign -fno-builtin-posix_memalign -fno-builtin-valloc -fno-builtin-pvalloc     -g -O2 -MT libtcmalloc_minimal_la-tcmalloc.lo -MD -MP -MF .deps/libtcmalloc_minimal_la-tcmalloc.Tpo -c -o libtcmalloc_minimal_la-tcmalloc.lo `test -f 'src/tcmalloc.cc' || echo './'`src/tcmalloc.cc
    libtool: compile:  g++ -DHAVE_CONFIG_H -I. -I./src -I./src -DNO_TCMALLOC_SAMPLES -D_THREAD_SAFE -DNDEBUG -Wall -Wwrite-strings -Woverloaded-virtual -Wno-sign-compare -fno-builtin-malloc -fno-builtin-free -fno-builtin-realloc -fno-builtin-calloc -fno-builtin-cfree -fno-builtin-memalign -fno-builtin-posix_memalign -fno-builtin-valloc -fno-builtin-pvalloc -g -O2 -MT libtcmalloc_minimal_la-tcmalloc.lo -MD -MP -MF .deps/libtcmalloc_minimal_la-tcmalloc.Tpo -c src/tcmalloc.cc  -fno-common -DPIC -o .libs/libtcmalloc_minimal_la-tcmalloc.o
    In file included from src/tcmalloc.cc:116:
    src/base/basictypes.h:349:5: error: #error Could not determine cache line length - unknown architecture
    src/base/basictypes.h: In constructor 'AssignAttributeStartEnd::AssignAttributeStartEnd(const char*, char**, char**)':
    src/base/basictypes.h:280: warning: '_dyld_present' is deprecated (declared at /usr/include/mach-o/dyld.h:237)
    src/base/basictypes.h:280: warning: '_dyld_present' is deprecated (declared at /usr/include/mach-o/dyld.h:237)
    make: *** [libtcmalloc_minimal_la-tcmalloc.lo] Error 1
    simon:gperftools-2.1 blyth$ 


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

gperftools-build(){
   gperftools-get
   gperftools-cd
   ./configure
   make
}


