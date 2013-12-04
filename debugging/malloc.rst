Malloc Debugging
==================

* http://stackoverflow.com/questions/2387289/can-you-recommend-a-good-debugging-malloc-library-for-linux


The GNU C library itself has some debugging features and hooks you can use to
add your own.

For documentation on a Linux system type ``info libc`` and then ``g Heap<TAB>``.
Another useful info node is "Hooks for Malloc", you can get there with
``g Hooks<TAB>``



* http://valgrind.org/docs/manual/QuickStart.html



* http://man7.org/linux/man-pages/man3/malloc.3.html

Recent versions of Linux libc (later than 5.4.23) and glibc (2.x)
include a malloc() implementation which is tunable via environment
variables.  For details, see mallopt(3).


::

    [blyth@belle7 debugging]$ MALLOC_CHECK_=1 ls
    malloc: using debugging hooks
    malloc.rst


Got some more info using ``MALLOC_CHECK_=1`` (continuing beyond the first corruption).

Now GDB fails::

    GiGaRunActionGDML::BeginOfRunAction i 0 c D
    FreeFilePath  return /data1/env/local/env/geant4/geometry/gdml/DVGX_20131203-2016/g4_00.dae i 1
    GiGaRunActionGDML::WriteDAE to /data1/env/local/env/geant4/geometry/gdml/DVGX_20131203-2016/g4_00.dae recreatePoly 0
    /data1/env/local/dyb/external/Python/2.7/i686-slc5-gcc41-dbg/bin/python: symbol lookup error: /data1/env/local/dyb/NuWa-trunk/lhcb/InstallArea/i686-slc5-gcc41-dbg/lib/libGaussTools.so: undefined symbol: _ZN10G4DAEWrite5WriteERK8G4StringPK15G4LogicalVolumeS2_ibb

    Program exited with code 0177.

    [blyth@belle7 DVGX_20131203-1719]$ c++filt _ZN10G4DAEWrite5WriteERK8G4StringPK15G4LogicalVolumeS2_ibb
    G4DAEWrite::Write(G4String const&, G4LogicalVolume const*, G4String const&, int, bool, bool)



Redirecting Malloc messages doesnt work
----------------------------------------

* http://stackoverflow.com/questions/7085360/squelching-glibc-memory-corruption-stack-trace-output

::

    The reason you can't redirect it is that it gets written specifically to
    /dev/tty, unless you have set the environment variable LIBC_FATAL_STDERR_. (I'm
    not sure this is documented anywhere, but the relevant code can be found here.)

gdb
-----

**single quotes** needed for GDB breakpoint tab completion

::

    (gdb) b 'G4VSceneHandler::G4VSceneHandler(G4VGraphicsSystem&, int, G4String const&)'                                                                                
    Breakpoint 2 at 0x7ed79b1: file src/G4VSceneHandler.cc, line 84.
    (gdb) 



Resolved by getting rid of fSummary in  G4VRML2FileSceneHandler.hh
------------------------------------------------------------------------

The culprit looks to be inconsisent class definition, due to not 
recompiling all the other classes that access this one.
Moral of the story:

* if possible avoid changing headers of classes under test when debugging those classes
* instead prefer to *take notes* in other debug only structures 


