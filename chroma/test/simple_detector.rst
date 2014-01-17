simple detector example
==========================

Runtime Mavericks ROOT/python issue with root-5.34.11
----------------------------------------------------------

::

    (chroma_env)delta:test blyth$ python -i 
    Python 2.7.6 (default, Nov 18 2013, 15:12:51) 
    [GCC 4.2.1 Compatible Apple LLVM 5.0 (clang-500.2.79)] on darwin
    Type "help", "copyright", "credits" or "license" for more information.
    >>> from chroma.io.root import RootWriter
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/usr/local/env/chroma_env/src/chroma/chroma/io/root.py", line 6, in <module>
        from chroma.rootimport import ROOT
      File "/usr/local/env/chroma_env/src/chroma/chroma/rootimport.py", line 6, in <module>
        import ROOT
      File "/usr/local/env/chroma_env/src/root-v5.34.11/lib/ROOT.py", line 257, in <module>
        _root.gPad         = _ExpandMacroFunction( "TVirtualPad",  "Pad" )
      File "/usr/local/env/chroma_env/src/root-v5.34.11/lib/ROOT.py", line 237, in __init__
        c = _root.MakeRootClass( klass )
    AttributeError: type object 'string' has no attribute 'c_str'
    >>> 

* http://root.cern.ch/phpBB3/viewtopic.php?f=14&t=17238

  * suggests this issue fixed after root-5.34.11 
  * TODO: try chroma-rebuild root with the lastest 5.34.12 
    added to chroma_pkgs


#. Moving to 5.34.14 avoids this issue, but get a freetype build issue, using 
   internal freetype in configure `chroma-kludge-root` avoids that. Fix
   is expected in as yet unreleased 5.34.15


Runtime hang, probably zmq ?
------------------------------

Suspect issue with zmq, starts 5 processes which dont take CPU 
and seems to hang.::

    blyth           39070   0.0  0.8  2778456 136152 s010  S+    2:06PM   0:02.19 python ./simple_detector.py
    blyth           39069   0.0  1.0  2798476 159416 s010  S+    2:06PM   0:02.01 python ./simple_detector.py
    blyth           39068   0.0  0.9  2796412 143680 s010  S+    2:06PM   0:01.73 python ./simple_detector.py
    blyth           39067   0.0  1.0  2804252 159648 s010  S+    2:06PM   0:01.96 python ./simple_detector.py
    blyth           39063   0.0  2.8 32545740 468452 s010  S+    2:06PM   0:05.30 python ./simple_detector.py

After 10 mins of seemingly hanging interrupted and then killed by closing tab::

    (chroma_env)delta:test blyth$ ./simple_detector.py 
    Info in <TUnixSystem::ACLiC>: creating shared library /Users/blyth/.chroma/root_C.so

    RooFit v3.59 -- Developed by Wouter Verkerke and David Kirkby 
                    Copyright (C) 2000-2013 NIKHEF, University of California & Stanford University
                    All rights reserved, please read http://roofit.sourceforge.net/license.txt


    ^CProcess G4GeneratorProcess-2:
    Process G4GeneratorProcess-3:
    Process G4GeneratorProcess-4:
    Process G4GeneratorProcess-1:
    Traceback (most recent call last):
    Traceback (most recent call last):
    Traceback (most recent call last):
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/multiprocessing/process.py", line 258, in _bootstrap
    Traceback (most recent call last):
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/multiprocessing/process.py", line 258, in _bootstrap
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/multiprocessing/process.py", line 258, in _bootstrap
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/multiprocessing/process.py", line 258, in _bootstrap
        self.run()
        self.run()
        self.run()
        self.run()
      File "/usr/local/env/chroma_env/src/chroma/chroma/generator/photon.py", line 32, in run
      File "/usr/local/env/chroma_env/src/chroma/chroma/generator/photon.py", line 32, in run
      File "/usr/local/env/chroma_env/src/chroma/chroma/generator/photon.py", line 32, in run
      File "/usr/local/env/chroma_env/src/chroma/chroma/generator/photon.py", line 32, in run
        ev = vertex_socket.recv_pyobj()
        ev = vertex_socket.recv_pyobj()
        ev = vertex_socket.recv_pyobj()
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/zmq/sugar/socket.py", line 344, in recv_pyobj
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/zmq/sugar/socket.py", line 344, in recv_pyobj
        ev = vertex_socket.recv_pyobj()
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/zmq/sugar/socket.py", line 344, in recv_pyobj
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/zmq/sugar/socket.py", line 344, in recv_pyobj
        s = self.recv(flags)
        s = self.recv(flags)
      File "socket.pyx", line 622, in zmq.backend.cython.socket.Socket.recv (zmq/backend/cython/socket.c:5403)
        s = self.recv(flags)
      File "socket.pyx", line 622, in zmq.backend.cython.socket.Socket.recv (zmq/backend/cython/socket.c:5403)
      File "socket.pyx", line 622, in zmq.backend.cython.socket.Socket.recv (zmq/backend/cython/socket.c:5403)
        s = self.recv(flags)
      File "socket.pyx", line 622, in zmq.backend.cython.socket.Socket.recv (zmq/backend/cython/socket.c:5403)
      File "socket.pyx", line 656, in zmq.backend.cython.socket.Socket.recv (zmq/backend/cython/socket.c:5222)
      File "socket.pyx", line 656, in zmq.backend.cython.socket.Socket.recv (zmq/backend/cython/socket.c:5222)
      File "socket.pyx", line 656, in zmq.backend.cython.socket.Socket.recv (zmq/backend/cython/socket.c:5222)
      File "socket.pyx", line 656, in zmq.backend.cython.socket.Socket.recv (zmq/backend/cython/socket.c:5222)
      File "socket.pyx", line 139, in zmq.backend.cython.socket._recv_copy (zmq/backend/cython/socket.c:1711)
      File "socket.pyx", line 139, in zmq.backend.cython.socket._recv_copy (zmq/backend/cython/socket.c:1711)
      File "checkrc.pxd", line 11, in zmq.backend.cython.checkrc._check_rc (zmq/backend/cython/socket.c:5713)
      File "checkrc.pxd", line 11, in zmq.backend.cython.checkrc._check_rc (zmq/backend/cython/socket.c:5713)
      File "socket.pyx", line 139, in zmq.backend.cython.socket._recv_copy (zmq/backend/cython/socket.c:1711)
      File "socket.pyx", line 139, in zmq.backend.cython.socket._recv_copy (zmq/backend/cython/socket.c:1711)
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/Geant4/__init__.py", line 242, in _run_abort
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/Geant4/__init__.py", line 242, in _run_abort
      File "checkrc.pxd", line 11, in zmq.backend.cython.checkrc._check_rc (zmq/backend/cython/socket.c:5713)
      File "checkrc.pxd", line 11, in zmq.backend.cython.checkrc._check_rc (zmq/backend/cython/socket.c:5713)
        raise KeyboardInterrupt
        raise KeyboardInterrupt
    KeyboardInterrupt
    KeyboardInterrupt
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/Geant4/__init__.py", line 242, in _run_abort
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/Geant4/__init__.py", line 242, in _run_abort
        raise KeyboardInterrupt
    KeyboardInterrupt
        raise KeyboardInterrupt
    KeyboardInterrupt



XQuartz DISPLAY issue
----------------------

* http://root.cern.ch/phpBB3/viewtopic.php?f=3&t=17240

Initially xclock and root fail to work saying::

    root: can't figure out DISPLAY, set it manually
    In case you run a remote ssh session, restart your ssh session with:
    =========>  ssh -Y

Resolved by logging out and back in again, following the XQuartz install.



zombie test.root file
-----------------------

::

    (chroma_env)delta:test blyth$ root test.root 
      *******************************************
      *                                         *
      *        W E L C O M E  to  R O O T       *
      *                                         *
      *   Version   5.34/14  16 December 2013   *
      *                                         *
      *  You are welcome to visit our Web site  *
      *          http://root.cern.ch            *
      *                                         *
      *******************************************

    ROOT 5.34/14 (v5-34-14@v5-34-14, Dec 16 2013, 12:23:58 on macosx64)

    CINT/ROOT C/C++ Interpreter version 5.18.00, July 2, 2010
    Type ? for help. Commands must be C++ statements.
    Enclose multiple statements between { }.
    root [0] 
    Attaching file test.root as _file0...
    Warning in <TFile::Init>: file test.root probably not closed, trying to recover
    Warning in <TFile::Init>: no keys recovered, file has been made a Zombie
    root [1] 
    root [1] 




