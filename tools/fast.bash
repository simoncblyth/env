# === func-gen- : tools/fast fgp tools/fast.bash fgn fast fgh tools
fast-src(){      echo tools/fast.bash ; }
fast-source(){   echo ${BASH_SOURCE:-$(env-home)/$(fast-src)} ; }
fast-vi(){       vi $(fast-source) ; }
fast-env(){      elocal- ; }
fast-usage(){ cat << EOU

FAST
=====

FAST is a set of tools for collecting, managing, and analyzing data about code performance.

* https://cdcvs.fnal.gov/redmine/projects/fast
* https://cdcvs.fnal.gov/redmine/attachments/5218/fast-concept.pdf
* https://cdcvs.fnal.gov/redmine/attachments/5219/fast-manual.pdf


BUILD on N
-----------

::

    sudo yum --enablerepo=epel install cmake

::

    [blyth@belle7 fastbuild]$ cmake ../fast
    ...
    -- Attempting to find 'iberty' using pkg-config
    -- checking for module 'iberty'
    --   package 'iberty' not found
    -- Searching system for 'iberty'
    --   iberty facility not found
           disabling bfd support
    --   bfd facility not found
    ...

::

    [blyth@belle7 fastbuild]$ sudo yum install binutils-devel


cmake
~~~~~~

::

    [blyth@belle7 fastbuild]$ cmake ../fast
    -- Located libunwind source files in /data1/env/local/env/tools/fast/libunwind
    -- Attempting to find 'iberty' using pkg-config
    -- checking for module 'iberty'
    --   package 'iberty' not found
    -- Searching system for 'iberty'
    -- Found iberty 
    libiberty found at /usr/lib
    -- Attempting to find 'bfd' using pkg-config
    -- checking for module 'bfd'
    --   package 'bfd' not found
    -- Searching system for 'bfd'
    -- Found bfd 
    -- Attempting to find 'rt' using pkg-config
    -- checking for module 'rt'
    --   package 'rt' not found
    -- Searching system for 'rt'
    -- Found rt 
    -- Configuring examples
    -- Configuring tests
    -- Configuring done
    -- Generating done
    -- Build files have been written to: /data1/env/local/env/tools/fastbuild
    [blyth@belle7 fastbuild]$ 


inline assembly problem
~~~~~~~~~~~~~~~~~~~~~~~~~~


Lack of a space and semicolon prevents compilation::

    /data1/env/local/env/tools/fast/SimpleProfiler/timing.h: In function 'void cpuid()':
    /data1/env/local/env/tools/fast/SimpleProfiler/timing.h:25: error: expected `)' before '::' token
    /data1/env/local/env/tools/fast/SimpleProfiler/timing.h:25: error: '__asm__volatile' was not declared in this scope
    /data1/env/local/env/tools/fast/SimpleProfiler/timing.h:26: error: expected `;' before '}' token

::

     23 static __inline__ void cpuid()
     24 {
     25   __asm__ volatile (".byte 0x0f, 0xa2":::"%eax", "%ebx", "%ecx", "%edx");
     26 }


::

    /usr/bin/c++   -DSimpleProfiler_EXPORTS -fPIC -I/data1/env/local/env/tools/fastbuild/libunwind/compiled/include -I/data1/env/local/env/tools/fast   -D_XOPEN_SOURCE -D__USE_POSIX199309 -O2 -g -DHAVE_RT -D_GNU_SOURCE -o CMakeFiles/SimpleProfiler.dir/SimpleProfiler/SimpleProfiler.cc.o -c /data1/env/local/env/tools/fast/SimpleProfiler/SimpleProfiler.cc
    /data1/env/local/env/tools/fast/SimpleProfiler/timing.h: In function 'uint64_t getCPUIDcycles()':
    /data1/env/local/env/tools/fast/SimpleProfiler/timing.h:25: error: PIC register '%ebx' clobbered in 'asm'
    /data1/env/local/env/tools/fast/SimpleProfiler/timing.h:25: error: PIC register '%ebx' clobbered in 'asm'




* http://www.ibiblio.org/gferg/ldp/GCC-Inline-Assembly-HOWTO.html







EOU
}
fast-dir(){ echo $(local-base)/env/tools/fast ; }
fast-cd(){  cd $(fast-dir); }
fast-mate(){ mate $(fast-dir) ; }
fast-get(){
   local dir=$(dirname $(fast-dir)) &&  mkdir -p $dir && cd $dir


   local url=https://cdcvs.fnal.gov/redmine/attachments/download/4734/fast.tar.gz
   local tgz=$(basename $url)

   [ ! -f "$tgz" ] && curl --insecure -L -O $url   # avoid SSL certificate issue with the --insecure
   [ ! -f "$tgz" ] && echo failed to download tgz $tgz && return 1 
   [ ! -d "fast" ] && mkdir fast && tar zxvf $tgz -C fast       # exploding tarball

}
