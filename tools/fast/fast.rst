
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


inline assembly compilation fPIC problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Lack of a space and semicolon prevents compilation::

    /data1/env/local/env/tools/fast/SimpleProfiler/timing.h: In function 'void cpuid()':
    /data1/env/local/env/tools/fast/SimpleProfiler/timing.h:25: error: expected `)' before '::' token
    /data1/env/local/env/tools/fast/SimpleProfiler/timing.h:25: error: '__asm__volatile' was not declared in this scope
    /data1/env/local/env/tools/fast/SimpleProfiler/timing.h:26: error: expected `;' before '}' token


Read timestampcounter into 64bit::

     16 static __inline__ uint64_t rdtsc()
     17 {
     18   uint64_t x;
     19   __asm__ volatile (".byte 0x0f, 0x31" : "=A" (x) );
     20   return x;
     21 }

     23 static __inline__ void cpuid()
     24 {
     25   __asm__ volatile (".byte 0x0f, 0xa2":::"%eax", "%ebx", "%ecx", "%edx");
     26 }


     95 uint64_t getCPUIDcycles()
     96 {
     97   uint64_t end, start;
     98   cpuid();
     99   start=rdtsc();
    100   cpuid();
    101   end=rdtsc();
    102   return end-start;
    103 }



* :google:`.byte 0x0f, 0xa2`

  * http://libquicktime.sourcearchive.com/documentation/2:1.1.5-1/cpuinfo_8c-source.html
  * http://en.wikipedia.org/wiki/CPUID  a way of interrogating the processor
  * http://gcc.gnu.org/bugzilla/show_bug.cgi?id=47602 
 

::

    /usr/bin/c++   -DSimpleProfiler_EXPORTS -fPIC -I/data1/env/local/env/tools/fastbuild/libunwind/compiled/include -I/data1/env/local/env/tools/fast   -D_XOPEN_SOURCE -D__USE_POSIX199309 -O2 -g -DHAVE_RT -D_GNU_SOURCE -o CMakeFiles/SimpleProfiler.dir/SimpleProfiler/SimpleProfiler.cc.o -c /data1/env/local/env/tools/fast/SimpleProfiler/SimpleProfiler.cc
    /data1/env/local/env/tools/fast/SimpleProfiler/timing.h: In function 'uint64_t getCPUIDcycles()':
    /data1/env/local/env/tools/fast/SimpleProfiler/timing.h:25: error: PIC register '%ebx' clobbered in 'asm'
    /data1/env/local/env/tools/fast/SimpleProfiler/timing.h:25: error: PIC register '%ebx' clobbered in 'asm'

Using `-fno-pic` rather than `fPIC` succeeds to compile::

    [blyth@belle7 fastbuild]$ /usr/bin/c++   -DSimpleProfiler_EXPORTS -fno-pic -I/data1/env/local/env/tools/fastbuild/libunwind/compiled/include -I/data1/env/local/env/tools/fast   -D_XOPEN_SOURCE -D__USE_POSIX199309 -O2 -g -DHAVE_RT -D_GNU_SOURCE -o CMakeFiles/SimpleProfiler.dir/SimpleProfiler/SimpleProfiler.cc.o -c /data1/env/local/env/tools/fast/SimpleProfiler/SimpleProfiler.cc


Inline Assembly Background
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://www.ibiblio.org/gferg/ldp/GCC-Inline-Assembly-HOWTO.html
* http://www.cs.virginia.edu/~evans/cs216/guides/x86.html
* http://www.ibm.com/developerworks/library/l-ia/index.html
* http://www.delorie.com/djgpp/doc/brennan/brennan_att_inline_djgpp.html



Register Juggling
~~~~~~~~~~~~~~~~~~~

* http://sourceforge.net/p/sevenzip/bugs/1271/
* http://newbiz.github.io/cpp/2010/12/20/Playing-with-cpuid.html


CPUID
~~~~~~

* http://datasheets.chipdb.org/Intel/x86/CPUID/24161821.pdf



Usage
-------

::

    [blyth@belle7 ~]$ profrun python -c "import time ; time.sleep(10) "

    profrun: subshell 7979 complete with status=0
    profrun: profrun of python complete;
    profrun: - profdata_7979_7979_543320563* files should exist.

    [blyth@belle7 ~]$ l profdata*
    -rwxrw-r-- 1 blyth blyth  148 Aug 14 18:52 profdata_7979_7979_543320563
    -rw-rw-r-- 1 blyth blyth  150 Aug 14 18:52 profdata_7979_7979_543320563_debugging
    -rw-rw-r-- 1 blyth blyth    4 Aug 14 18:52 profdata_7979_7979_543320563_libraries
    -rw-rw-r-- 1 blyth blyth 3178 Aug 14 18:52 profdata_7979_7979_543320563_maps
    -rw-rw-r-- 1 blyth blyth 2208 Aug 14 18:52 profdata_7979_7979_543320563_names
    -rw-rw-r-- 1 blyth blyth   96 Aug 14 18:52 profdata_7979_7979_543320563_paths
    -rw-rw-r-- 1 blyth blyth  415 Aug 14 18:52 profdata_7979_7979_543320563_timing
    -rw-rw-r-- 1 blyth blyth   93 Aug 14 18:52 profdata_7979_7979_543320563_totals
    [blyth@belle7 ~]$ 



