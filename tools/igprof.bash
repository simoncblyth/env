# === func-gen- : tools/igprof fgp tools/igprof.bash fgn igprof fgh tools src base/func.bash
igprof-source(){   echo ${BASH_SOURCE} ; }
igprof-edir(){ echo $(dirname $(igprof-source)) ; }
igprof-ecd(){  cd $(igprof-edir); }
igprof-dir(){  echo $LOCAL_BASE/env/tools/igprof ; }
igprof-cd(){   cd $(igprof-dir); }
igprof-vi(){   vi $(igprof-source) ; }
igprof-env(){  elocal- ; }
igprof-usage(){ cat << EOU

IgProf
========

* https://igprof.org/install.html


CentOS7 install
---------------

sudo curl -o /etc/yum.repos.d/igprof.repo https://bintray.com/igprof/slc7_x86-64-test/rpm

## sudo yum update   # chickened out if this, far too many updates
sudo yum install igprof

::

    [blyth@localhost env]$ rpm -ql igprof
    /usr/bin/igprof
    /usr/bin/igprof-analyse
    /usr/bin/igprof-analyse-old
    /usr/bin/igprof-func
    /usr/bin/igprof-navigator
    /usr/bin/igprof-navigator-summary
    /usr/bin/igprof-populator
    /usr/bin/igprof-segment
    /usr/bin/igprof-symbol-sizes
    /usr/bin/igpython-analyse
    /usr/bin/igtrace
    /usr/bin/igtrace-mmap-analysis
    /usr/bin/igtrace-mmap-summary
    /usr/include/igprof/sym-resolve.h
    /usr/lib/libigprof.so
    [blyth@localhost env]$ 



Running
---------

* https://igprof.org/running.html


-pp/-mp 
     performance/memory profiler
-t  
     pick executable name to profile, otherwise all spawned processes 


::

    tmp ; igprof -d -mp TCURANDTest

    [blyth@localhost opticks]$ l igprof*
    -rw-rw-r--. 1 blyth blyth 104878 Jul 10 16:28 igprof.TCURANDTest.122589.1562747315.807789.gz
    -rw-rw-r--. 1 blyth blyth 122417 Jul 10 16:28 igprof.python.122712.1562747315.750446.gz
    -rw-rw-r--. 1 blyth blyth 122362 Jul 10 16:28 igprof.python.122655.1562747313.920678.gz
    -rw-rw-r--. 1 blyth blyth 122367 Jul 10 16:28 igprof.python.122622.1562747312.880984.gz



Analysis
----------

* https://igprof.org/analysis.html








EOU
}
igprof-get(){
   local dir=$(dirname $(igprof-dir)) &&  mkdir -p $dir && cd $dir

}
