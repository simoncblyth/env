NuWa Profiling with FAST SimpleProfiler
=========================================

Appears stuck rather early::

    RootIOCnvSvc                       WARNING  no "default" input file name specified, unregistered TES paths will not be saved.
    StaticCalibDataSvc                    INFO User specified PmtDataFile is []
    StaticCalibDataSvc                    INFO Loading Master.pmtCalibMap.txt
    StaticCalibDataSvc                    INFO Loading Master.feeCalibMap.txt
    StaticSimDataSvc                      INFO Opened input file: /data1/env/local/dyb/NuWa-trunk/dybgaudi/DataModel/DataSvc/share/pmtDataTable.txt
    StaticSimDataSvc                      INFO Opening input file: /data1/env/local/dyb/NuWa-trunk/dybgaudi/DataModel/DataSvc/share/feeDataTable.txt


Attaching to hung process::

    blyth    24390  0.2  3.4 203480 141844 pts/1   S+   12:43   0:04 python opw-sim.py
    [blyth@belle7 ~]$ 
    [blyth@belle7 ~]$ 
    [blyth@belle7 ~]$ gdb $(which python) 24390
    GNU gdb Red Hat Linux (6.5-25.el5rh)
    Copyright (C) 2006 Free Software Foundation, Inc.
    GDB is free software, covered by the GNU General Public License, and you are
    welcome to change it and/or distribute copies of it under certain conditions.
    Type "show copying" to see the conditions.
    There is absolutely no warranty for GDB.  Type "show warranty" for details.
    This GDB was configured as "i386-redhat-linux-gnu"...Using host libthread_db library "/lib/libthread_db.so.1".

    Attaching to program: /data1/env/local/dyb/external/Python/2.7/i686-slc5-gcc41-dbg/bin/python, process 24390

Suspect a deadlock::

    Loaded symbols for /data1/env/local/dyb/NuWa-trunk/../external/XercesC/2.8.0/i686-slc5-gcc41-dbg/lib/libxerces-depdom.so.28
    0x00335410 in __kernel_vsyscall ()
    (gdb) bt

    #0  0x00335410 in __kernel_vsyscall ()
    #1  0x00b626e9 in __lll_lock_wait () from /lib/libpthread.so.0
    #2  0x00b5dd9f in _L_lock_885 () from /lib/libpthread.so.0
    #3  0x00b5dc66 in pthread_mutex_lock () from /lib/libpthread.so.0
    #4  0x0533b1ae in std::locale::locale () from /usr/lib/libstdc++.so.6
    #5  0x0533663c in std::ios_base::ios_base$base () from /usr/lib/libstdc++.so.6
    #6  0x05377252 in std::basic_ostringstream<char, std::char_traits<char>, std::allocator<char> >::basic_ostringstream () from /usr/lib/libstdc++.so.6
    #7  0x008c1cca in SimpleProfiler::findInvalidRanges (this=0x950220) at /data1/env/local/env/tools/fast/SimpleProfiler/SimpleProfiler.cc:919
    #8  0x008c293b in SimpleProfiler::stacktrace (this=0x950220, addresses=0x8c69008, nmax=1000) at /data1/env/local/env/tools/fast/SimpleProfiler/SimpleProfiler.cc:464
    #9  0x008c3386 in fastSigFunc () at /data1/env/local/env/tools/fast/SimpleProfiler/SimpleProfiler.cc:163
    #10 <signal handler called>
    #11 0x053998ed in __gnu_cxx::__atomic_add () from /usr/lib/libstdc++.so.6
    #12 0x05337db0 in std::ios_base::_M_init () from /usr/lib/libstdc++.so.6
    #13 0x0534b3c8 in std::basic_ios<char, std::char_traits<char> >::init () from /usr/lib/libstdc++.so.6
    #14 0x05379129 in std::basic_stringstream<char, std::char_traits<char>, std::allocator<char> >::basic_stringstream () from /usr/lib/libstdc++.so.6
    #15 0xb484245e in StaticSimDataSvc::initialize (this=0xa9dc6a8) at ../src/components/StaticSimDataSvc.cc:115
    #16 0x035f1879 in Service::sysInitialize (this=0xa9dc6a8) at ../src/Lib/Service.cpp:71
    #17 0xb54bea5d in ServiceManager::initializeServices (this=0xb5663450) at ../src/ApplicationMgr/ServiceManager.cpp:381
    #18 0xb5272734 in ApplicationMgr::initialize (this=0xb5662f60) at ../src/ApplicationMgr/ApplicationMgr.cpp:556
    #19 0x03dcdee6 in method_4812 (retaddr=0xa8e0100, o=0xb5663390) at ../i686-slc5-gcc41-dbg/dict/GaudiKernel/dictionary_dict.cpp:10987
    #20 0x002deadd in ROOT::Cintex::Method_stub_with_context (context=0xa766540, result=0xaba3384, libp=0xaba33dc) at cint/cintex/src/CINTFunctional.cxx:319
    #21 0x045ac034 in ?? ()
    #22 0x0a766540 in ?? ()
    #23 0x0aba3384 in ?? ()
    #24 0x00000000 in ?? ()


::

    [blyth@belle7 dybgaudi]$ vi DataModel/DataSvc/src/components/StaticSimDataSvc.cc +115

    102       msgStr << "Opening input file: " << m_feeDataFileName;
    103       msg->reportMessage("StaticSimDataSvc",MSG::INFO,msgStr.str());
    104     }
    105     int channelId;
    106     std::string description;
    107     double channelThreshold, adcRangeHigh, adcRangeLow;
    108     double adcBaselineHigh, adcBaselineLow;
    109     std::string line;
    110     while( std::getline(input,line) ){
    111       std::ostringstream msgStr;
    112       msgStr << "Got line: " << line;
    113       msg->reportMessage("StaticSimDataSvc",MSG::VERBOSE,msgStr.str());
    114       msgStr.str("");
    115       std::stringstream streamLine(line);
    116       if( streamLine.peek() == '#' ){
    117     continue;
    118       }
    119       streamLine >> channelId >> description
    120          >> channelThreshold >> adcRangeHigh >> adcRangeLow
    121          >> adcBaselineHigh >> adcBaselineLow;
    122 

::

    [blyth@belle7 fast]$ vi /data1/env/local/env/tools/fast/SimpleProfiler/SimpleProfiler.cc +919

     915 void SimpleProfiler::findInvalidRanges()
     916 {
     917 //    static int callcount=1;
     918     pid_t pid = getpid();
     919     std::ostringstream strinput;
     920     strinput << "/proc/" << pid << "/maps";
     921     std::ifstream input(strinput.str().c_str());
     922     std::string line;
     923     unexecutable.clear(); //We have to read the whole maps file anyway, since
     924                           //  [vdso] may be at the end.  Best to just 
     925                           //  rederive the NX regions rather than attempting
     926                           //  a diff-type approach.  
     927     unsigned long int lastUpperBound=0, temp_start, temp_end;
     928     while (getline(input, line))
     929     {
     930 //       fprintf(stderr, line.c_str());
     931        //[vdso] is strangely given permissions '---p' on systems with static
     932        //  vsyscall rather than a true DSO, as in some x86_64 systems, 


* :google:`ostringstream deadlock`
* http://stackoverflow.com/questions/3707138/why-ostringstream-could-not-work-well-in-multithreading-environment
* https://bugs.launchpad.net/ubuntu/+source/gcc-4.3/+bug/453518  gcc bug fixed from 4.5
* http://gcc.gnu.org/bugzilla/show_bug.cgi?id=40088

::

    [blyth@belle7 fast]$ gcc -v
    Using built-in specs.
    Target: i386-redhat-linux
    Configured with: ../configure --prefix=/usr --mandir=/usr/share/man --infodir=/usr/share/info --enable-shared --enable-threads=posix --enable-checking=release --with-system-zlib --enable-__cxa_atexit --disable-libunwind-exceptions --enable-libgcj-multifile --enable-languages=c,c++,objc,obj-c++,java,fortran,ada --enable-java-awt=gtk --disable-dssi --enable-plugin --with-java-home=/usr/lib/jvm/java-1.4.2-gcj-1.4.2.0/jre --with-cpu=generic --host=i386-redhat-linux
    Thread model: posix
    gcc version 4.1.2 20080704 (Red Hat 4.1.2-46)





Deadlock occurs at different places on each run, in the below at reading in the muon file::

    (gdb) bt
    #0  0x00ebe410 in __kernel_vsyscall ()
    #1  0x00b626e9 in __lll_lock_wait () from /lib/libpthread.so.0
    #2  0x00b5dd9f in _L_lock_885 () from /lib/libpthread.so.0
    #3  0x00b5dc66 in pthread_mutex_lock () from /lib/libpthread.so.0
    #4  0x0533b1ae in std::locale::locale () from /usr/lib/libstdc++.so.6
    #5  0x0533663c in std::ios_base::ios_base$base () from /usr/lib/libstdc++.so.6
    #6  0x05377252 in std::basic_ostringstream<char, std::char_traits<char>, std::allocator<char> >::basic_ostringstream () from /usr/lib/libstdc++.so.6
    #7  0x00c38cca in SimpleProfiler::findInvalidRanges (this=0xcc7220) at /data1/env/local/env/tools/fast/SimpleProfiler/SimpleProfiler.cc:919
    #8  0x00c3993b in SimpleProfiler::stacktrace (this=0xcc7220, addresses=0x86df008, nmax=1000) at /data1/env/local/env/tools/fast/SimpleProfiler/SimpleProfiler.cc:464
    #9  0x00c3a386 in fastSigFunc () at /data1/env/local/env/tools/fast/SimpleProfiler/SimpleProfiler.cc:163
    #10 <signal handler called>
    #11 0x00b5dc60 in pthread_mutex_lock () from /lib/libpthread.so.0
    #12 0x0533b1ae in std::locale::locale () from /usr/lib/libstdc++.so.6
    #13 0x05337db0 in std::ios_base::_M_init () from /usr/lib/libstdc++.so.6
    #14 0x0534b3c8 in std::basic_ios<char, std::char_traits<char> >::init () from /usr/lib/libstdc++.so.6
    #15 0x05378c4e in std::basic_istringstream<char, std::char_traits<char>, std::allocator<char> >::basic_istringstream () from /usr/lib/libstdc++.so.6
    #16 0x067468ae in HepEvt2HepMC::fill (this=0xb1cd928, source_desc=0xa4e1a7c "/data1/env/local/env/muon_simulation/optical_photon_weighting/OPW/tenthousandmuons") at ../src/components/HepEvt2HepMC.cc:155
    #17 0x06723642 in GtHepEvtGenTool::mutate (this=0xa73d190, event=@0xac9d808) at ../src/components/GtHepEvtGenTool.cc:51
    #18 0x067141a0 in GtGenerator::execute (this=0xa7323a8) at ../src/components/GtGenerator.cc:69
    #19 0x03d16408 in Algorithm::sysExecute (this=0xa7323a8) at ../src/Lib/Algorithm.cpp:558
    #20 0x046c8d4e in DybBaseAlg::sysExecute (this=0xa7323a8) at ../src/lib/DybBaseAlg.cc:53
    #21 0x03d9241a in MinimalEventLoopMgr::executeEvent (this=0xa481980) at ../src/Lib/MinimalEventLoopMgr.cpp:450
    #22 0x05dfa956 in DybEventLoopMgr::executeEvent (this=0xa481980, par=0x0) at ../src/DybEventLoopMgr.cpp:125
    #23 0x05dfb18a in DybEventLoopMgr::nextEvent (this=0xa481980, maxevt=10) at ../src/DybEventLoopMgr.cpp:188
    #24 0x03d90dbd in MinimalEventLoopMgr::executeRun (this=0xa481980, maxevt=10) at ../src/Lib/MinimalEventLoopMgr.cpp:400
    #25 0xb506e6d9 in ApplicationMgr::executeRun (this=0xb543c2f8, evtmax=10) at ../src/ApplicationMgr/ApplicationMgr.cpp:867
    #26 0x049eef57 in method_3426 (retaddr=0xb1cd080, o=0xb543c724, arg=@0xa1b4158) at ../i686-slc5-gcc41-dbg/dict/GaudiKernel/dictionary_dict.cpp:4375
    #27 0x008e1add in ROOT::Cintex::Method_stub_with_context (context=0xa1b4150, result=0xacabd8c, libp=0xacabde4) at cint/cintex/src/CINTFunctional.cxx:319
    #28 0x04578034 in ?? ()
    #29 0x0a1b4150 in ?? ()
    #30 0x0acabd8c in ?? ()
    #31 0x00000000 in ?? ()


Conclusion FAST SimpleProfiler is not usable with gcc41, try again with gcc45 or later. Hmm this might cause 
a problem wirth gperftools too.



