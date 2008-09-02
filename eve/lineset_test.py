
"""

== Interactive running : ipython and python -i behave the same ==

 ipython lineset_test.py
(Bool_t)(1)
Python 2.5.1 (r251:54863, Sep  2 2008, 13:30:44) 
Type "copyright", "credits" or "license" for more information.

IPython 0.8.4 -- An enhanced Interactive Python.
?         -> Introduction and overview of IPython's features.
%quickref -> Quick reference.
help      -> Python's own help system.
object?   -> Details about 'object'. ?object also works, ?? prints more.

In [1]: TGLLogicalShape::Draw:0: RuntimeWarning: display-list registration failed.



==  Non-interactive ==


    python lineset_test.py
*** glibc detected *** double free or corruption (out): 0x0a797b50 ***
Aborted 

   
==  gdb run, putting break on abort ==


   gdb $(which python) 
GNU gdb Red Hat Linux (6.3.0.0-1.159.el4rh)
Copyright 2004 Free Software Foundation, Inc.
GDB is free software, covered by the GNU General Public License, and you are
welcome to change it and/or distribute copies of it under certain conditions.
Type "show copying" to see the conditions.
There is absolutely no warranty for GDB.  Type "show warranty" for details.
This GDB was configured as "i386-redhat-linux-gnu"...Using host libthread_db library "/lib/tls/libthread_db.so.1".

(gdb) b abort
Function "abort" not defined.
Make breakpoint pending on future shared library load? (y or [n]) y
Breakpoint 1 (abort) pending.
(gdb) r lineset_test.py
Starting program: /data/env/system/python/Python-2.5.1/bin/python lineset_test.py
[Thread debugging using libthread_db enabled]
[New Thread -1208518976 (LWP 8425)]
Breakpoint 2 at 0x844196
Pending breakpoint "abort" resolved
Detaching after fork from child process 8428.
[New Thread -1211655264 (LWP 8433)]
[Thread -1211655264 (LWP 8433) exited]
*** glibc detected *** double free or corruption (out): 0x096a9d00 ***
[Switching to Thread -1208518976 (LWP 8425)]

Breakpoint 2, 0x00844196 in abort () from /lib/tls/libc.so.6
(gdb) info stack
#0  0x00844196 in abort () from /lib/tls/libc.so.6
#1  0x00876cca in __libc_message () from /lib/tls/libc.so.6
#2  0x0087d55f in _int_free () from /lib/tls/libc.so.6
#3  0x0087d93a in free () from /lib/tls/libc.so.6
#4  0x0586ab31 in operator delete () from /usr/lib/libstdc++.so.6
#5  0x00bd62f1 in TStorage::ObjectDealloc () from /data/env/local/root/root_v5.21.02/root/lib/libCore.so
#6  0x00bb68f7 in TObject::operator delete () from /data/env/local/root/root_v5.21.02/root/lib/libCore.so
#7  0x00bf6e4a in TTimer::~TTimer$delete () from /data/env/local/root/root_v5.21.02/root/lib/libCore.so
#8  0x00c15950 in TCollection::GarbageCollect () from /data/env/local/root/root_v5.21.02/root/lib/libCore.so
#9  0x00c1f3e6 in TOrdCollection::Delete () from /data/env/local/root/root_v5.21.02/root/lib/libCore.so
#10 0x00be6c26 in TSystem::~TSystem$base () from /data/env/local/root/root_v5.21.02/root/lib/libCore.so
#11 0x00c5d7d3 in TUnixSystem::~TUnixSystem$delete () from /data/env/local/root/root_v5.21.02/root/lib/libCore.so
#12 0x00bce70a in TROOT::~TROOT () from /data/env/local/root/root_v5.21.02/root/lib/libCore.so
#13 0x00bce733 in __tcf_0 () from /data/env/local/root/root_v5.21.02/root/lib/libCore.so
#14 0x00845597 in exit () from /lib/tls/libc.so.6
#15 0x0082fded in __libc_start_main () from /lib/tls/libc.so.6
#16 0x08048501 in _start ()
(gdb) 






"""



## translated from $ROOTSYS/tutorials/eve/lineset_test.C to demo  this issue 

import ROOT

def lineset_test( nlines = 40, nmarkers = 4):
    ROOT.TEveManager.Create()
    r = ROOT.TRandom(0)
    s = 100

    ls = ROOT.TEveStraightLineSet()

    for i in range(nlines):
        ls.AddLine( r.Uniform(-s,s), r.Uniform(-s,s), r.Uniform(-s,s) ,
                    r.Uniform(-s,s), r.Uniform(-s,s), r.Uniform(-s,s))
        nm = int(nmarkers*r.Rndm())
        for m in range(nm):
            ls.AddMarker( i, r.Rndm() )
    ls.SetMarkerSize(1.5)
    ls.SetMarkerStyle(4)

    ROOT.gEve.AddElement(ls)
    ROOT.gEve.Redraw3D()
    return ls

if __name__=='__main__':
    lineset_test()
