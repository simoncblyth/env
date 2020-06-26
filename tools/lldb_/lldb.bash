# === func-gen- : tools/lldb/lldb fgp tools/lldb/lldb.bash fgn lldb fgh tools/lldb
lldb-src(){      echo tools/lldb/lldb.bash ; }
lldb-source(){   echo ${BASH_SOURCE:-$(env-home)/$(lldb-src)} ; }
lldb-vi(){       vi $(lldb-source) ; }
lldb-env(){      elocal- ; }
lldb-usage(){ cat << EOU

LLDB Experience
=================

Breakpoints
-------------

::

    (lldb) br lis
    (lldb) br dis 1
    1 breakpoints disabled.
    (lldb) br en 1
    1 breakpoints enabled.


Avoid quit confirmation
-------------------------

::

    settings show  # list all setting 
    settings set interpreter.prompt-on-quit false 


init script ~/.lldbinit
---------------------------

epsilon:~ blyth$ cat ~/.lldbinit

    settings set interpreter.prompt-on-quit false 


Batch Mode one liners
----------------------

Source list from backtrace address 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    PATH=/usr/bin lldb $(which CerenkovMinimal) -o "source list -a 0x10002160e"  --batch


See::

    lldb --help

The PATH is set to avoid having macports python in the PATH which gives an error
"ImportError: cannot import name _remove_dead_weakref"::

    epsilon:~ blyth$ PATH=/usr/bin lldb $(which CerenkovMinimal) -o "source list -a 0x10002160e"  --batch

    (lldb) target create "/usr/local/opticks/lib/CerenkovMinimal"
    Current executable set to '/usr/local/opticks/lib/CerenkovMinimal' (x86_64).
    (lldb) source list -a 0x10002160e
    /usr/local/opticks/lib/CerenkovMinimal`L4Cerenkov::PostStepDoIt(G4Track const&, G4Step const&) + 2683 at /Users/blyth/opticks/examples/Geant4/CerenkovMinimal/L4Cerenkov.cc:352
       341 	#ifdef WITH_OPTICKS_ALIGN
       342 	        G4Opticks::GetOpticks()->setAlignIndex(i); 
       343 	#endif
       344 	
       345 			G4double rand;
       346 			G4double sampledEnergy, sampledRI; 
       347 			G4double cosTheta, sin2Theta;
       348 			
       349 			// sample an energy
       350 	
       351 			do {
    -> 352 				rand = G4UniformRand();	
       353 				sampledEnergy = Pmin + rand * dp; 
       354 				sampledRI = Rindex->Value(sampledEnergy);
       355 				cosTheta = BetaInverse / sampledRI;  
       356 	
       357 				sin2Theta = (1.0 - cosTheta)*(1.0 + cosTheta);
    epsilon:~ blyth$ 





Source list
-------------

* https://stackoverflow.com/questions/18112842/how-can-i-find-the-address-of-a-stack-trace-in-lldb-for-ios  

::

   (lldb) source list -a 0x000000010b99b2c2


    ::

    (lldb) p (char*)main
    (char *) $6 = 0x0000000100011970 "UH\xffffff89\xffffffe5H\xffffff81\xffffffec\xffffff90\x01"
    (lldb) source list -a 0x0000000100011970
    /usr/local/opticks/lib/CerenkovMinimal`main at /Users/blyth/opticks/examples/Geant4/CerenkovMinimal/CerenkovMinimal.cc:6
    -> 6   	{
       7   	    OPTICKS_LOG(argc, argv); 
       8   	
       9   	    CMixMaxRng mmr;  // switch engine to a instrumented shim, to see the random stream
       10  	
       11  	    G4 g(1) ; 
    (lldb) 

::

    (lldb) source list -a (char*)main+552
    /usr/local/opticks/lib/CerenkovMinimal`main + 552 at /Users/blyth/opticks/examples/Geant4/CerenkovMinimal/CerenkovMinimal.cc:11
       3   	#include "CMixMaxRng.hh"
       4   	
       5   	int main(int argc, char** argv)
       6   	{
       7   	    OPTICKS_LOG(argc, argv); 
       8   	
       9   	    CMixMaxRng mmr;  // switch engine to a instrumented shim, to see the random stream
       10  	
    -> 11  	    G4 g(1) ; 
       12  	    return 0 ; 
       13  	}
       14  	
       15  	
    (lldb) 


Seems that the address incorporates the offset already::

    8   libG4event.dylib                    0x00000001022c571a G4EventManager::DoProcessing(G4Event*)                                                               + 3306     
    9   libG4event.dylib                    0x00000001022c6c2f G4EventManager::ProcessOneEvent(G4Event*)                                                            + 47       
    10  libG4run.dylib                      0x00000001021d29f5 G4RunManager::ProcessOneEvent(int)                                                                   + 69       
    11  libG4run.dylib                      0x00000001021d2825 G4RunManager::DoEventLoop(int, char const*, int)                                                     + 101      
    12  libG4run.dylib                      0x00000001021d0ce1 G4RunManager::BeamOn(int, char const*, int)                                                          + 193      
    13  CerenkovMinimal                     0x0000000100032dcd G4::beamOn(int)                                                                                      + 45       
    14  CerenkovMinimal                     0x0000000100032c77 G4::G4(int)                                                                                          + 1015     
    15  CerenkovMinimal                     0x0000000100032dfb G4::G4(int)                                                                                          + 27       
    16  CerenkovMinimal                     0x0000000100011ba2 main + 562
    17  libdyld.dylib                       0x00007fff7acac015 start + 1



/tmp/simstream.txt::

    30 :   0.406647 :      + 2662 L4Cerenkov::PostStepDoIt(G4Track const&, G4Step const&)
    31 :   0.490262 :      + 2883 L4Cerenkov::PostStepDoIt(G4Track const&, G4Step const&)
    32 :   0.671936 :      + 2978 L4Cerenkov::PostStepDoIt(G4Track const&, G4Step const&)

    (lldb) b "L4Cerenkov::PostStepDoIt(G4Track const&, G4Step const&)"
    Breakpoint 1: where = CerenkovMinimal`L4Cerenkov::PostStepDoIt(G4Track const&, G4Step const&) + 27 at L4Cerenkov.cc:197, address = 0x0000000100020bbb
    (lldb) 


    (lldb) source list -a 0x0000000100020bbb
    /usr/local/opticks/lib/CerenkovMinimal`L4Cerenkov::PostStepDoIt(G4Track const&, G4Step const&) + 27 at /Users/blyth/opticks/examples/Geant4/CerenkovMinimal/L4Cerenkov.cc:197
       188 	// segment and uniformly azimuth w.r.t. the particle direction. The 
       189 	// parameters are then transformed into the Master Reference System, and 
       190 	// they are added to the particle change. 
       191 	
       192 	{
       193 		//////////////////////////////////////////////////////
       194 		// Should we ensure that the material is dispersive?
       195 		//////////////////////////////////////////////////////
       196 	
    -> 197 	        aParticleChange.Initialize(aTrack);
       198 	
       199 	        const G4DynamicParticle* aParticle = aTrack.GetDynamicParticle();
       200 	        const G4Material* aMaterial = aTrack.GetMaterial();
       201 	
       202 		G4StepPoint* pPreStepPoint  = aStep.GetPreStepPoint();
    (lldb) 


image lookup
---------------


::

    (lldb) image lookup -v --address 0x0000000100020bbb
          Address: CerenkovMinimal[0x0000000100020bbb] (CerenkovMinimal.__TEXT.__text + 62027)
          Summary: CerenkovMinimal`L4Cerenkov::PostStepDoIt(G4Track const&, G4Step const&) + 27 at L4Cerenkov.cc:197
           Module: file = "/usr/local/opticks/lib/CerenkovMinimal", arch = "x86_64"
      CompileUnit: id = {0x00000000}, file = "/Users/blyth/opticks/examples/Geant4/CerenkovMinimal/L4Cerenkov.cc", language = "c++"
         Function: id = {0x400059edd}, name = "PostStepDoIt", range = [0x0000000100020ba0-0x0000000100022033)
         FuncType: id = {0x400059edd}, decl = L4Cerenkov.hh:129, compiler_type = "class G4VParticleChange *(const class G4Track &, const class G4Step &)"
           Blocks: id = {0x400059edd}, range = [0x100020ba0-0x100022033)
        LineEntry: [0x0000000100020bbb-0x0000000100020bc2): /Users/blyth/opticks/examples/Geant4/CerenkovMinimal/L4Cerenkov.cc:197:9
           Symbol: id = {0x000003db}, range = [0x0000000100020ba0-0x0000000100022040), name="L4Cerenkov::PostStepDoIt(G4Track const&, G4Step const&)", mangled="_ZN10L4Cerenkov12PostStepDoItERK7G4TrackRK6G4Step"
         Variable: id = {0x400059ef7}, name = "this", type = "L4Cerenkov *", location =  DW_OP_fbreg(-104), decl = 
         Variable: id = {0x400059f05}, name = "aTrack", type = "const G4Track &", location =  DW_OP_fbreg(-112), decl = L4Cerenkov.cc:183
         Variable: id = {0x400059f14}, name = "aStep", type = "const G4Step &", location =  DW_OP_fbreg(-120), decl = L4Cerenkov.cc:183
         Variable: id = {0x400059f23}, name = "aParticle", type = "const G4DynamicParticle *", location =  DW_OP_fbreg(-128), decl = L4Cerenkov.cc:199
         Variable: id = {0x400059f32}, name = "aMaterial", type = "const G4Material *", location =  DW_OP_fbreg(-136), decl = L4Cerenkov.cc:200
         Variable: id = {0x400059f41}, name = "pPreStepPoint", type = "G4StepPoint *", location =  DW_OP_fbreg(-144), decl = L4Cerenkov.cc:202









adding python breakpoint func
--------------------------------

Little information provided by google, perhaps because the help is informative::


    (lldb) help br com add
         Add a set of commands to a breakpoint, to be executed whenever the breakpoint is hit.

    Syntax: breakpoint command add <cmd-options> <breakpt-id>

    ...

       -F <python-function> ( --python-function <python-function> )
            Give the name of a Python function to run as command for this breakpoint. Be sure to give a module name if appropriate.




Setting Environment
---------------------

::

     (lldb) env DYLD_INSERT_LIBRARIES=/usr/lib/libgmalloc.dylib
     (lldb) env OPTICKS_QUERY_LIVE="range:0:1"

lldbinit
-----------

Check the file::

    command source ~/.lldbinit


numpy into system python for use from lldb ?
-----------------------------------------------

::

    delta:ana blyth$ /usr/bin/python -i
    Python 2.7.5 (default, Mar  9 2014, 22:15:05) 
    [GCC 4.2.1 Compatible Apple LLVM 5.0 (clang-500.0.68)] on darwin
    Type "help", "copyright", "credits" or "license" for more information.
    >>> 
    >>> print "\n".join(sys.path)

    /Users/blyth
    /System/Library/Frameworks/Python.framework/Versions/2.7/lib/python27.zip
    /System/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7
    /System/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/plat-darwin
    /System/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/plat-mac
    /System/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/plat-mac/lib-scriptpackages
    /System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python
    /System/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/lib-tk
    /System/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/lib-old
    /System/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/lib-dynload
    /System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/PyObjC
    /Library/Python/2.7/site-packages
    >>> 
    delta:ana blyth$ cat /Library/Python/2.7/site-packages/README 
    This directory exists so that 3rd party packages can be installed
    here.  Read the source for site.py for more details.
    delta:ana blyth$ 




Introspection
---------------

Look at members of a base class::

    (lldb) p m_photons
    (NPY<float> *) $3 = 0x0000000109001840

    (lldb) p *m_photons
    (NPY<float>) $4 = {}

    (lldb) p *(NPYBase*)m_photons
    (NPYBase) $7 = {
      m_dim = 3
      m_ni = 0
      m_nj = 4
      m_nk = 4
      m_nl = 0
      m_sizeoftype = '\x04'
      m_type = FLOAT
      m_buffer_id = -1
      m_buffer_target = -1
      m_aux = 0x0000000000000000
      m_verbose = false
      m_allow_prealloc = false
      m_shape = size=3 {
        [0] = 0
        [1] = 4
        [2] = 4
      }
      m_metadata = "{}"
      m_has_data = true
      m_dynamic = true
    }




lldb logging
--------------

* https://stackoverflow.com/questions/37296802/how-to-log-to-the-console-in-an-lldb-data-formatter



Experience with lldb python scripting
----------------------------------------

Not all variables available, use "frame variables" to find some, 
even "this" sometimes not available.

::

    (lldb) frame variable
    (G4int) moduloFactor = <no location, value may have been optimized out>
    (G4Transportation *) this = 0x0000000110226ae0


    (lldb) help br com add

    (lldb) script

    >>> help(lldb.frame)
    >>> help(lldb.SBValue)  # eg this

    (lldb) script
    Python Interactive Interpreter. To exit, type 'quit()', 'exit()' or Ctrl-D.
    >>> print lldb.frame
    frame #0: 0x000000010528fd8b libG4tracking.dylib`G4SteppingManager::DefinePhysicalStepLength(this=0x0000000111a550b0) + 299 at G4SteppingManager2.cc:181

    >>> this = lldb.frame.FindVariable("this")

    >>> print "\n".join([str(this.GetChildAtIndex(_)) for _ in range(this.GetNumChildren())])

    (G4bool) KillVerbose = true
    (G4UserSteppingAction *) fUserSteppingAction = 0x000000010d369370
    (G4VSteppingVerbose *) fVerbose = 0x0000000111a55440
    (G4double) PhysicalStep = 1.7976931348623157E+308
    ...

    >>> memb_ = lambda this:"\n".join([str(this.GetChildAtIndex(_)) for _ in range(len(this))])

    >>> print memb_(this)

    (G4bool) KillVerbose = true
    (G4UserSteppingAction *) fUserSteppingAction = 0x000000010d369370
    (G4VSteppingVerbose *) fVerbose = 0x0000000111a55440
    ...

    >>> p = this.GetChildMemberWithName("fCurrentProcess")
    >>> print p
    (G4VProcess *) fCurrentProcess = 0x000000010d3ad880

    >>> re.compile("\"(\S*)\"").search(str(this.GetValueForExpressionPath(".fCurrentProcess.theProcessName"))).group(1)
    'OpBoundary'



    

Python Scripting
------------------




* https://llvm.org/svn/llvm-project/lldb/trunk/include/lldb/API/SBValue.h

* https://stackoverflow.com/questions/41089145/how-do-i-access-c-array-float-values-from-lldb-python-api

* https://www.codesd.com/item/how-do-i-access-the-floating-point-values-in-table-c-of-the-lldb-python-api.html

* http://idrisr.com/2015/10/12/debugging-a-debugger.html

* https://github.com/llvm-mirror/lldb/tree/master/packages/Python/lldbsuite/test/functionalities

* https://lldb.llvm.org/python_reference/index.html

* https://lldb.llvm.org/varformats.html

::

    (lldb) type summary add -P Rectangle
    Enter your Python command(s). Type 'DONE' to end.
    def function (valobj,internal_dict):
        height_val = valobj.GetChildMemberWithName('height')
        width_val = valobj.GetChildMemberWithName('width')
        height = height_val.GetValueAsUnsigned(0)
        width = width_val.GetValueAsUnsigned(0)
        area = height*width
        perimeter = 2*(height + width)
        return 'Area: ' + str(area) + ', Perimeter: ' + str(perimeter)
        DONE
    (lldb) frame variable
    (Rectangle) r1 = Area: 20, Perimeter: 18
    (Rectangle) r2 = Area: 72, Perimeter: 36
    (Rectangle) r3 = Area: 16, Perimeter: 16


::

    >>> print lldb.target
    OKG4Test
    >>> print dir(lldb.target)




EOU
}
lldb-dir(){ echo $(env-home)/tools/lldb_ ; }
lldb-cd(){  cd $(lldb-dir); }
lldb-c(){  cd $(lldb-dir); }


lldb-i(){

    lldb-c
   /usr/bin/python -i standalone.py

}


lldb-ckm()
{
    local addr=${1:-0x10002160e}
    PATH=/usr/bin lldb $(which CerenkovMinimal) -o "source list -a $addr -c 5"  --batch
}




