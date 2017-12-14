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


Setting Environment
---------------------

::

     (lldb) env DYLD_INSERT_LIBRARIES=/usr/lib/libgmalloc.dylib

lldbinit
-----------

Check the file::

    command source ~/.lldbinit






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






adding python breakpoint func
--------------------------------

(lldb) help br com add
     Add a set of commands to a breakpoint, to be executed whenever the breakpoint is hit.

Syntax: breakpoint command add <cmd-options> <breakpt-id>



EOU
}
lldb-dir(){ echo $(env-home)/tools/lldb_ ; }
lldb-cd(){  cd $(lldb-dir); }
lldb-c(){  cd $(lldb-dir); }


lldb-i(){

    lldb-c
   /usr/bin/python -i standalone.py

}

