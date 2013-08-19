fast-src(){      echo tools/fast/fast.bash ; }
fast-source(){   echo ${BASH_SOURCE:-$(env-home)/$(fast-src)} ; }
fast-vi(){       vi $(fast-source) ; }
fast-env(){      elocal- ; fast-setup ;  }
fast-usage(){ cat << EOU

Moved to :e:`tools/fast`


Profiling causing segmentations
---------------------------------

::

    [blyth@belle7 20130816-1642]$ ./opw-sim.py > opw-sim.log


Lower level running
----------------------

::

    [blyth@belle7 20130816-1642]$ fast-on
    === fast-on : CAUTION setting LD_PRELOAD envvar /data1/env/local/env/tools/fast/build/proftools/bin/../../SimpleProfiler/Linux.i686/libSimpleProfiler.so : all commands will be profiles until : fast-off
    [blyth@belle7 20130816-1642]$ python opw-sim.py


libunwind
----------

* http://www.nongnu.org/libunwind/
* http://www.nongnu.org/libunwind/man/libunwind(3).html

call unw_getcontext() to get a snapshot of the CPU registers (machine-state).
Then you initialize an unwind cursor based on this snapshot. This is done with
a call to unw_init_local(). The cursor now points to the current frame, that
is, the stack frame that corresponds to the current activation of function F().
The unwind cursor can then be moved ``up'' (towards earlier stack frames) by
calling unw_step(). By repeatedly calling this routine, you can uncover the
entire call-chain that led to the activation of function F(). A positive return
value from unw_step() indicates that there are more frames in the chain, zero
indicates that the end of the chain has been reached, and any negative value
indicates that some sort of error has occurred.


SimpleProfiler collection done by `SimpleProfiler::stacktrace`::

     393 // Record stack frame instruction pointers into 'addresses'.  At most
     394 // nmax entries will be filled.  Return the observed stack depth;
     395 // negative values indicate an error.
     396 int SimpleProfiler::stacktrace (address addresses[], int nmax)
     397 {
     398   ++samples_total_;
     399   int  depth = 0;
     400   // We don't actually care about this address -- so record a zero instead.
     401   // if (depth < nmax) addresses[depth++] = &SimpleProfiler::stacktrace;
     402   addresses[depth++] = 0;
     403 
     404   // Get the current machine state.
     405   unw_context_t uc;
     406   int rc = unw_getcontext(&uc);
     407 
     ...



HOW IT COLLECTS SAMPLES
------------------------

Once loaded by the operating systems dynamic loader, libSimpleProfiler registers a signal 
handler to respond to the SIGPROF signal, and then sets up an interval timer to send 
the SIGPROF signal every ten milliseconds. Each call to the registered signal handler 
captures a single sample. 
A sample consists of a series of memory addresses which make up the call stack, 
the location in memory to which each called function will return when that function is 
completed. These samples are buffered in memory, and when a sufficient number have 
been recorded, they are written to the raw data file described in section 7.5. 
When the programs main function exits, the raw data files written during data collec- 
tion are processed, and the several output files described in section 7 are written.


::

    [blyth@belle7 SimpleProfiler]$ grep SIGPROF *.cc
    SigMaskHandler.cc:      if (how==SIG_BLOCK || how==SIG_SETMASK) sigdelset((sigset_t*)set,SIGPROF);
    SimpleProfiler.cc:// The signal handler. We register this to handle the SIGPROF signal.
    SimpleProfiler.cc:      // signals that occur while another SIGPROF is being handled.
    SimpleProfiler.cc:    int mysig = SIGPROF;
    [blyth@belle7 SimpleProfiler]$ 



ISSUES
-------------

* http://indico.cern.ch/getFile.py/access?contribId=4&sessionId=0&resId=0&materialId=slides&confId=216064

Geant4 people looking at other profilers


hostid fails
~~~~~~~~~~~~~~~~

Cannot open processes `os.popen` eg::

    [blyth@belle7 20130816-1642]$ python check_hostid.py 
    1087078072
    [blyth@belle7 20130816-1642]$ 
    [blyth@belle7 20130816-1642]$ 
    [blyth@belle7 20130816-1642]$ fast-on
    === fast-on : CAUTION setting LD_PRELOAD envvar /data1/env/local/env/tools/fast/build/proftools/bin/../../SimpleProfiler/Linux.i686/libSimpleProfiler.so : all commands will be profiles until : fast-off
    [blyth@belle7 20130816-1642]$ python check_hostid.py 
    Warning, failed to get hostid directly, error message was ` invalid literal for int() with base 16: '0x' ` now try to get hostid from hostname.
    1087078072
    The function '_Z41__static_initialization_and_destruction_0ii' in library 'libGaudiKernelDict.so' called from 'libMathCore.so:G__cpp_setup_tagtableG__MathCore', 'libMathCore.so:_GLOBAL__I__ZN4ROOT20GenerateInitInstanceEPK7TRandom', 'libGaudiPythonDict.so:_GLOBAL__I__ZN84_GLOBAL__N_.._i686_slc5_gcc41_dbg_dict_GaudiPython_kernel_dict.cpp_00000000_B74175634nsb0E', 'libGaudiKernelDict.so:_GLOBAL__I__ZN88_GLOBAL__N_.._i686_slc5_gcc41_dbg_dict_GaudiKernel_dictionary_dict.cpp_FB02ED6B_A0BA49BF4nsb0E', 'libCore.so:_ZN5TCint27UpdateListOfGlobalFunctionsEv' appears to violate the 'One Definition Rule'.
    It is defined at addresses 0xeeee72, 0x5e0c3c2.
    The function '_Z41__static_initialization_and_destruction_0ii' in library 'libGaudiPythonDict.so' called from 'libMathCore.so:G__cpp_setup_tagtableG__MathCore', 'libMathCore.so:_GLOBAL__I__ZN4ROOT20GenerateInitInstanceEPK7TRandom', 'libGaudiPythonDict.so:_GLOBAL__I__ZN84_GLOBAL__N_.._i686_slc5_gcc41_dbg_dict_GaudiPython_kernel_dict.cpp_00000000_B74175634nsb0E', 'libGaudiKernelDict.so:_GLOBAL__I__ZN88_GLOBAL__N_.._i686_slc5_gcc41_dbg_dict_GaudiKernel_dictionary_dict.cpp_FB02ED6B_A0BA49BF4nsb0E', 'libCore.so:_ZN5TCint27UpdateListOfGlobalFunctionsEv' appears to violate the 'One Definition Rule'.
    It is defined at addresses 0xeeee72, 0x375c3f2.
    The function '_Z41__static_initialization_and_destruction_0ii' in library 'libMathCore.so' called from 'libMathCore.so:G__cpp_setup_tagtableG__MathCore', 'libMathCore.so:_GLOBAL__I__ZN4ROOT20GenerateInitInstanceEPK7TRandom', 'libGaudiPythonDict.so:_GLOBAL__I__ZN84_GLOBAL__N_.._i686_slc5_gcc41_dbg_dict_GaudiPython_kernel_dict.cpp_00000000_B74175634nsb0E', 'libGaudiKernelDict.so:_GLOBAL__I__ZN88_GLOBAL__N_.._i686_slc5_gcc41_dbg_dict_GaudiKernel_dictionary_dict.cpp_FB02ED6B_A0BA49BF4nsb0E', 'libCore.so:_ZN5TCint27UpdateListOfGlobalFunctionsEv' appears to violate the 'One Definition Rule'.
    It is defined at addresses 0xeeee72, 0x1862d08.
    The function '__tcf_0' in library 'libCore.so' called from 'libc-2.5.so:__GI_exit' appears to violate the 'One Definition Rule'.
    It is defined at addresses 0x6971de, 0xeea5d6.
    The function '__tcf_0' in library 'libReflex.so' called from 'libc-2.5.so:__GI_exit' appears to violate the 'One Definition Rule'.
    It is defined at addresses 0x6971de, 0xc285c0
    [blyth@belle7 20130816-1642]$ fast-off
    [blyth@belle7 20130816-1642]$ cat check_hostid.py 

    from DybPython.hostid import hostid
    print hostid()
     

Failing to cat the muons
~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    10401 ApplicationMgr                        INFO Application Manager Initialized successfully
    10402 ApplicationMgr                        INFO Application Manager Started successfully
    10403 ToolSvc.catHepEvt                     INFO Filled HEPEvt cache with 0 events
    10404 IssueLogger                          FATAL FATAL  ../src/components/HepEvt2HepMC.cc:32  "No cached events to return, call fill()"
    10405 #1  0x693fdb0 HepEvt2HepMC::generate(HepMC::GenEvent*&)  [/data1/env/local/dyb/NuWa-trunk/dybgaudi/InstallArea/i686-slc5-gcc41-dbg/lib/libGenTools.so]
    10406 #2  0x691c556 GtHepEvtGenTool::mutate(HepMC::GenEvent&)  [/data1/env/local/dyb/NuWa-trunk/dybgaudi/InstallArea/i686-slc5-gcc41-dbg/lib/libGenTools.so]
    10407 #3  0x690cebe GtGenerator::execute()  [/data1/env/local/dyb/NuWa-trunk/dybgaudi/InstallArea/i686-slc5-gcc41-dbg/lib/libGenTools.so]
    10408 #4  0x4a2e21a Algorithm::sysExecute()  [/data1/env/local/dyb/NuWa-trunk/gaudi/InstallArea/i686-slc5-gcc41-dbg/lib/libGaudiKernel.so]
    10409 #5  0x3654cbc DybBaseAlg::sysExecute()  [/data1/env/local/dyb/NuWa-trunk/dybgaudi/InstallArea/i686-slc5-gcc41-dbg/lib/libDybAlgLib.so]
    10410 #6  0x4aaa128 MinimalEventLoopMgr::executeEvent(void*)  [/data1/env/local/dyb/NuWa-trunk/gaudi/InstallArea/i686-slc5-gcc41-dbg/lib/libGaudiKernel.so]
    10411 #7  0x5485880 DybEventLoopMgr::executeEvent(void*)  [/data1/env/local/dyb/NuWa-trunk/dybgaudi/InstallArea/i686-slc5-gcc41-dbg/lib/libDybEventMgr.so]
    10412 #8  0x5485a2a DybEventLoopMgr::nextEvent(int)  [/data1/env/local/dyb/NuWa-trunk/dybgaudi/InstallArea/i686-slc5-gcc41-dbg/lib/libDybEventMgr.so]
    10413 #9  0x4aa8b84 MinimalEventLoopMgr::executeRun(int)  [/data1/env/local/dyb/NuWa-trunk/gaudi/InstallArea/i686-slc5-gcc41-dbg/lib/libGaudiKernel.so]
    10414 #10 0x8d9261e ApplicationMgr::executeRun(int)  [/data1/env/local/dyb/NuWa-trunk/gaudi/InstallArea/i686-slc5-gcc41-dbg/lib/libGaudiSvc.so]


Failing to `GtHepEvtGenTool::mutate`, presumably from another popen::

    00050         m_parser = new HepEvt2HepMC;
    00051         if (m_parser->fill(m_source.c_str()).isFailure()) {
    00052             fatal () << "Failed to fill primary vertices using \""
    00053                      << m_source << "\"" << endreq;
    00054             return StatusCode::FAILURE;
    00055         }
    00056         info () << "Filled HEPEvt cache with " << m_parser->cacheSize()
    00057                 << " events" << endreq;

* http://dayabay.bnl.gov/dox/GenTools/html/GtHepEvtGenTool_8cc_source.html
* http://dayabay.bnl.gov/dox/GenTools/html/HepEvt2HepMC_8cc.html#a8c3f05b6bd785a33c90bf640b7d073be

Try workaround of changing source to the absolute path to the muon file rather than the cat command with a pipe "|"



EOU
}
fast-dir(){ echo $(local-base)/env/tools/fast ; }
fast-bin(){ echo $(fast-dir)/$(fast-build-name)/proftools/bin ; }
fast-lib(){ echo $(fast-bin)/../../SimpleProfiler/$(uname -sm | sed 's/ /./')/libSimpleProfiler.so ; } 
fast-on(){
   local msg="=== $FUNCNAME :"
   echo $msg CAUTION setting LD_PRELOAD envvar $(fast-lib) : all commands will be profiles until : fast-off 
   export LD_PRELOAD=$(fast-lib)  ;
}
fast-off(){
   unset LD_PRELOAD
}

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

fast-setup(){
   local bin=$(fast-bin)
   [ -d "$bin" ] && export PATH=$bin:$PATH
}
  

fast-build-name(){ echo build ; }

fast-cmake(){
   fast-cd
   local bld=$(fast-build-name)
   mkdir -p $bld 
   cd $bld 
   cmake ../

}

fast-fix(){
   # add space and semicolon
   perl -pi -e 's,(\s*__asm__)(volatile)(.*\)),$1 $2$3;, && print ' ../SimpleProfiler/timing.h 
}




