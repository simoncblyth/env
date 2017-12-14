#!/usr/bin/python
"""
NB thats the system python, not the macports one

* https://lldb.llvm.org/python-reference.html

::

    /usr/bin/python -i standalone.py 

    >>> print thread.GetFrameAtIndex(2)
    frame #2: 0x0000000107d59ff3 standalone`main(argc=1, argv=0x00007fff57ea6f38) + 83 at standalone.cc:28

"""

import os, sys
sys.path.append("/Library/Developer/CommandLineTools/Library/PrivateFrameworks/LLDB.framework/Resources/Python")
from collections import OrderedDict
import lldb

from env.tools.lldb_.evaluate import evaluate_frame, evaluate_var, evaluate_obj

nam = "standalone"
exe = "/tmp/%s" % nam
src = "%s.cc" % nam
cmd = "cc %(src)s -g -lc++ -o %(exe)s " % locals() # without -g fails to find breakpoints
print "compiling : %s " % cmd 
rc = os.system(cmd)
assert rc == 0, rc

def find_bpline(path, mkr ):
    marker = "// (*lldb*) %s" % mkr
    nls = filter( lambda nl:nl[1].find(marker) > -1, enumerate(file(path).readlines()) )
    assert len(nls) == 1
    return int(nls[0][0])+1, nls[0][1]


debugger = lldb.SBDebugger.Create()
debugger.SetAsync (False)

target = debugger.CreateTargetWithFileAndArch (exe, lldb.LLDB_ARCH_DEFAULT)
print "target:", target
assert target

filename = target.GetExecutable().GetFilename()
print "filename ", filename

bpln, bpline = find_bpline(src, "Exit")
print "bpline:%s" % bpline
bp = target.BreakpointCreateByLocation(src, bpln )  
print bp

process = target.LaunchSimple (None, None, os.getcwd())   # synchronous mode returns at bp 
print "process:", process
assert process

state = process.GetState ()
print "state:", state
assert state == lldb.eStateStopped

thread = process.GetThreadAtIndex (0)
print "thread:", thread
assert thread

frame = thread.GetFrameAtIndex (0)
assert frame
print "frame", frame

function = frame.GetFunction()
assert function
print "function:", function
    

error = lldb.SBError()
ef = evaluate_frame(frame, vdump=True, error=error)
print ef 
print ef["o"]["_s"]

