# === func-gen- : tools/gdb fgp tools/gdb.bash fgn gdb fgh tools src base/func.bash
gdb-source(){   echo ${BASH_SOURCE} ; }
gdb-edir(){ echo $(dirname $(gdb-source)) ; }
gdb-ecd(){  cd $(gdb-edir); }
gdb-dir(){  echo $LOCAL_BASE/env/tools/gdb ; }
gdb-cd(){   cd $(gdb-dir); }
gdb-vi(){   vi $(gdb-source) ; }
gdb-env(){  elocal- ; }
gdb-usage(){ cat << EOU

GDB
====

Refs
-----

* https://sourceware.org/gdb/onlinedocs/gdb/Set-Breaks.html


How to inform GDB of a change in source tree path ?
----------------------------------------------------

* https://ftp.gnu.org/old-gnu/Manuals/gdb/html_node/gdb_48.html


* "directory" seems not to work ?


Wrong paths in backtrace::

    Program received signal SIGABRT, Aborted.
    0x00007ffff652e387 in raise () from /lib64/libc.so.6
    (gdb) bt
    #0  0x00007ffff652e387 in raise () from /lib64/libc.so.6
    #1  0x00007ffff652fa78 in abort () from /lib64/libc.so.6
    #2  0x00007ffff65271a6 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007ffff6527252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007ffff7b39f7d in SCurandState::RngMax (
        path=0x421ab0 "/afs/ihep.ac.cn/users/b/blyth/.opticks/rngcache/RNG/QCurandState_3000000_0_0.bin")
        at /home/simon/opticks/sysrap/SCurandState.cc:80
    #5  0x00007ffff7b39e37 in SCurandState::RngMax () at /home/simon/opticks/sysrap/SCurandState.cc:71
    #6  0x00007ffff7b39a03 in SCurandState::Desc[abi:cxx11]() () at /home/simon/opticks/sysrap/SCurandState.cc:17
    #7  0x0000000000403b6e in main (argc=1, argv=0x7fffffffc7b8) at /home/simon/opticks/sysrap/tests/SCurandStateTest.cc:21
    (gdb) 



::

    L7[blyth@lxslc710 blyth]$ cat fixlib.sh 
    #!/bin/bash -l 
    src=/home/simon
    dst=/home/blyth
    cd $OPTICKS_RELEASE_PREFIX/lib64
    for path in $(ls -1 *.so) ; do 
        echo $path
        sed -i -e "s,$src,$dst,g" $path 
    done
    L7[blyth@lxslc710 blyth]$ 


::

    Program received signal SIGABRT, Aborted.
    0x00007ffff652e387 in raise () from /lib64/libc.so.6
    (gdb) bt
    #0  0x00007ffff652e387 in raise () from /lib64/libc.so.6
    #1  0x00007ffff652fa78 in abort () from /lib64/libc.so.6
    #2  0x00007ffff65271a6 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007ffff6527252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007ffff7b39f7d in SCurandState::RngMax (
        path=0x421ab0 "/afs/ihep.ac.cn/users/b/blyth/.opticks/rngcache/RNG/QCurandState_3000000_0_0.bin")
        at /home/blyth/opticks/sysrap/SCurandState.cc:80
    #5  0x00007ffff7b39e37 in SCurandState::RngMax () at /home/blyth/opticks/sysrap/SCurandState.cc:71
    #6  0x00007ffff7b39a03 in SCurandState::Desc[abi:cxx11]() () at /home/blyth/opticks/sysrap/SCurandState.cc:17
    #7  0x0000000000403b6e in main (argc=1, argv=0x7fffffffc818) at /home/blyth/opticks/sysrap/tests/SCurandStateTest.cc:21
    (gdb) 


Could arrange for the source build to think 
its building into a tree with root::

   /afs/ihep.ac.cn/users/b/blyth 



Building On GDB : eg for stack sampling to find hotspots
----------------------------------------------------------

* ``--interpreter=mi`` GDB/MI is a line based machine oriented text interface to GDB 
* https://sourceware.org/gdb/onlinedocs/gdb/GDB_002fMI.html#GDB_002fMI
* https://github.com/jasonrohrer/wallClockProfiler/blob/master/wallClockProfiler.cpp


::

   execlp( "gdb", "gdb", "-nx", "--interpreter=mi", progName, NULL );

::

    1664 
    1665     int usPerSample = lrint( 1000000 / samplesPerSecond );
    1666 
    1668     printf( "Sampling %.2f times per second, for %d usec between samples\n",
    1669             samplesPerSecond, usPerSample );
    ...
    1684     while( !programExited &&
    1685            ( detatchSeconds == -1 ||
    1686              time( NULL ) < startTime + detatchSeconds ) ) {
    1687         usleep( usPerSample );
    1688 
    1689         // interrupt
    1690         if( inNumArgs == 3 ) {
    1691             // we ran our program with run above to redirect output
    1692             // thus -exec-interrupt won't work
    1693             log( "Sending SIGINT to target process", inArgs[2] );
    1694 
    1695             kill( pid, SIGINT );
    1696             }
    1697         else {
    1698             sendCommand( "-exec-interrupt" );
    1699             }
    1700 
    1701         waitForGDBInterruptResponse();
    1702 
    1704         if( !programExited ) {
    1705             // sample stack
    1706             sendCommand( "-stack-list-frames" );
    1707             logGDBStackResponse();
    1708             numSamples++;
    1709             }
    1710 
    1711         if( !programExited ) {
    1712             // continue running
    1713 
    1714             sendCommand( "-exec-continue" );
    1715             skipGDBResponse();
    1716             }
    1717         }


* https://softwarerecs.stackexchange.com/questions/51826/wall-clock-profiler-for-linux





            

:google:`GDB stack sampler using interpreter=mi`

* https://stackoverflow.com/questions/16771393/writing-front-end-for-gdb
* https://sourceware.org/gdb/wiki/GDB%20Front%20Ends 



* https://github.com/gperftools/gperftools



Thin GDB/MI wrappers
~~~~~~~~~~~~~~~~~~~~~~

* https://github.com/search?q=GDB%2FMI

* https://github.com/search?q=gdb+mi

* https://github.com/cs01/pygdbmi
* https://github.com/AntyMew/libmigdb


:google:`GDB/MI profiler`

* https://github.com/copy/gdbprofiler

* https://github.com/search?q=gdb+profile&type=Repositories

* http://poormansprofiler.org/

* https://github.com/Muon/gdbprof

* https://github.com/markhpc/gdbpmp

* https://fritshoogland.wordpress.com/2020/01/04/advanced-usage-of-gdb-for-profiling/



ignoring breakpoint for a count
---------------------------------

* https://stackoverflow.com/questions/2956889/how-to-make-a-gdb-breakpoint-only-break-after-the-point-is-reached-a-given-numbe

::

    (gdb) ignore 1 30
    Will ignore next 30 crossings of breakpoint 1.



breakpoint basics
--------------------

* add : b 101    # line num
* listing : "i b" or "info b" 
* d 1            # delete breakpoint 1 


dprintf 
--------

The dynamic printf command dprintf combines a breakpoint with formatted
printing of your program’s data to give you the effect of inserting printf
calls into your program on-the-fly, without having to recompile it.

* https://doc.ecoscentric.com/gnutools/doc/gdb/Dynamic-Printf.html



Printing something on hitting breakpoint
------------------------------------------ 

* https://stackoverflow.com/questions/6517423/how-to-do-an-specific-action-when-a-certain-breakpoint-is-hit-in-gdb

After creating a breakpoint, use "commands"::


    (gdb) b 512
    Breakpoint 2 at 0x7fffd09bba7a: file ../src/DsG4Scintillation.cc, line 512.
    (gdb) commands
    Type commands for breakpoint(s) 2, one per line.
    End with a line saying just "end".
    >silent
    >print ancestor
    >cont
    >end
    (gdb) 






Avoid the annoying quit confirmation
--------------------------------------

* https://stackoverflow.com/questions/4355978/get-rid-of-quit-anyway-prompt-using-gdb-just-kill-the-process-and-quit

~/.gdbinit::

   define hook-quit
      set confirm off
   end


Avoid confirming pending breakpoints
---------------------------------------

::

    set breakpoint pending on



Scripted setting of multiple breakpoints
------------------------------------------

::

     44 tds(){
     ..
     59    local script=$JUNOTOP/offline/Examples/Tutorial/share/tut_detsim.py
     60    local args="--opticks --no-guide_tube gun"
     61    #local args="--opticks gun"  
     62 
     63    if [ -z "$BP" ]; then
     64       H="" 
     65       B=""         
     66       T="-ex r"
     67    else
     68       H="-ex \"set breakpoint pending on\" "
     69       B="" 
     70       for bp in $BP ; do
     71          B="$B -ex \"break $bp\" "
     72       done
     73       T="-ex \"info break\" -ex r"
     74    fi 
     75    
     76    local runline="gdb $H $B $T --args python $script $args $* "
     77    echo $runline
     78    eval $runline
     79 }  
     80 
     81 tds0(){ BP=DetSim0Svc::createDetectorConstruction tds ; }
     82 tds1(){ BP=R12860PMTManager::R12860PMTManager     tds ; }
     83 tds2(){ BP="R12860PMTManager::R12860PMTManager DetSim0Svc::createDetectorConstruction"   tds ; }
     84 




Breakpoints
--------------

* plant a std::raise(SIGINT) at strategic point where symbols will be available, 
  and switch that on with an option, eg --cg4sigint 

* TAB completion of symbolic bp requires all namespaces 

::

    (gdb) b 'HepRandom::put'
    Function "HepRandom::put" not defined.
    Make breakpoint pending on future shared library load? (y or [n]) n


    (gdb) b "CLHEP::HepRandom::put( <TAB>    DOES NOT WORK WITH DOUBLE QUOTES        

    (gdb) b 'CLHEP::HepRandom::put( <TAB>    THIS WORKS : BUT FINDS WRONG SYMBOL

    (gdb) b 'CLHEP::HepRandom::put(std::ostream&) const' 



    (gdb) b /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/externals/clhep/src/RandomEngine.cc:58
    Breakpoint 1 at 0x7fffe7b4d46e: file /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/externals/clhep/src/RandomEngine.cc, line 58.





EOU
}
gdb-get(){
   local dir=$(dirname $(gdb-dir)) &&  mkdir -p $dir && cd $dir

}
