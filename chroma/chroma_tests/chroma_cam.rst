bin/chroma-cam
================

solution to pygame segfault : dont fork
----------------------------------------

Added option for that::

    chroma-cam -F @chroma.models.lionsolid


pygame parachute segmentation fault
-------------------------------------

::

    (chroma_env)delta:chroma blyth$ chroma-cam @chroma.models.lionsolid
    INFO:chroma:Flattening detector mesh...
    INFO:chroma:  triangles: 74358
    INFO:chroma:  vertices:  37181
    INFO:chroma:Building new BVH using recursive grid algorithm.
    Expanding 250 parent nodes
    Merging 74358 nodes to 19472 parents
    Merging 19729 nodes to 5469 parents
    Merging 5469 nodes to 1378 parents
    Merging 1378 nodes to 340 parents
    Merging 340 nodes to 72 parents
    Merging 72 nodes to 21 parents
    Merging 21 nodes to 6 parents
    Merging 6 nodes to 2 parents
    Merging 2 nodes to 1 parent
    INFO:chroma:BVH generated in 0.7 seconds.
    INFO:chroma:Saving BVH (5bb27feefcb2dda31256bf6e755cb451:default) to cache.
    Fatal Python error: (pygame parachute) Segmentation Fault
    (chroma_env)delta:chroma blyth$ 
    (chroma_env)delta:chroma blyth$ 
    (chroma_env)delta:chroma blyth$ chroma-cam @chroma.models.lionsolid
    INFO:chroma:Flattening detector mesh...
    INFO:chroma:  triangles: 74358
    INFO:chroma:  vertices:  37181
    INFO:chroma:Loading BVH "default" for geometry from cache.
    Fatal Python error: (pygame parachute) Segmentation Fault
    (chroma_env)delta:chroma blyth$ 

Refering to models by path has the same behavior::

    (chroma_env)delta:chroma blyth$ chroma-cam chroma/models/liberty.stl 
    INFO:chroma:Flattening detector mesh...
    INFO:chroma:  triangles: 17060
    INFO:chroma:  vertices:  8573
    INFO:chroma:Building new BVH using recursive grid algorithm.
    Expanding 217 parent nodes
    Merging 17060 nodes to 5589 parents
    Expanding 19 parent nodes
    Merging 5944 nodes to 1729 parents
    Merging 1751 nodes to 475 parents
    Merging 475 nodes to 109 parents
    Merging 109 nodes to 32 parents
    Merging 32 nodes to 7 parents
    Merging 7 nodes to 2 parents
    Merging 2 nodes to 1 parent
    INFO:chroma:BVH generated in 0.7 seconds.
    INFO:chroma:Saving BVH (c61432f54300d20b1b936b38da17a7e6:default) to cache.
    INFO:chroma:loaded geometry <chroma.geometry.Geometry object at 0x1112261d0> 
    INFO:chroma:starting view [1024, 576] 
    INFO:chroma:create Camera 
    INFO:chroma:Camera.__init__
    INFO:chroma:Camera.__init__ done
    INFO:chroma:start Camera 
    INFO:chroma:join Camera 
    Fatal Python error: (pygame parachute) Segmentation Fault
    (chroma_env)delta:chroma blyth$ 




Why use multiprocessing, there is only one camera ? One viewer ?

* http://kivy.org/docs/faq.html
* http://stackoverflow.com/questions/13218362/fatal-python-error-pygame-parachute-segmentation-fault-when-changing-window


issue avoided by not forking
------------------------------

* http://stackoverflow.com/questions/14719065/pycuda-multiprocessing-issue-on-os-x-10-8
* http://bugs.python.org/issue8713



Use my added `view_nofork` or `Camera._run()` that doesnt invoke
`multiprocessing.run` which does the fork succeeds to run a display the model::

    (chroma_env)delta:tests blyth$ chroma-cam @chroma.models.lionsolid
    INFO:chroma:Flattening detector mesh...
    INFO:chroma:  triangles: 74358
    INFO:chroma:  vertices:  37181
    INFO:chroma:Loading BVH "default" for geometry from cache.
    INFO:chroma:loaded geometry <chroma.geometry.Geometry object at 0x11659f650> 
    INFO:chroma:starting view [1024, 576] 
    INFO:chroma:create Camera 
    INFO:chroma:Camera.__init__
    INFO:chroma:Camera.__init__ done
    INFO:chroma:_run Camera 
    INFO:chroma:Optimization: Sufficient memory to move triangles onto GPU
    INFO:chroma:Optimization: Sufficient memory to move vertices onto GPU
    INFO:chroma:device usage:
    ----------
    nodes           101.4K   1.6M
    total                    1.6M
    ----------
    device total             2.1G
    device used            231.5M
    device free              1.9G


lldb not informative
------------------------------

Trying to find a stack trace.

* http://lldb.llvm.org/lldb-gdb.html
* how to jump process in lldb ?

Hmm as using multiprocessing not easy to debug::

    (chroma_env)delta:bin blyth$ lldb -- `which python` /usr/local/env/chroma_env/src/chroma/bin/chroma-cam @chroma.models.lionsolid --debug
    Current executable set to '/usr/local/env/chroma_env/bin/python' (x86_64).
    (lldb) r
    Process 53332 launched: '/usr/local/env/chroma_env/bin/python' (x86_64)
    Process 53332 stopped and restarted: thread 1 received signal: SIGCHLD
    INFO:chroma:Flattening detector mesh...
    INFO:chroma:  triangles: 74358
    INFO:chroma:  vertices:  37181
    INFO:chroma:Loading BVH "default" for geometry from cache.
    INFO:chroma:loaded geometry <chroma.geometry.Geometry object at 0x10a698bd0> 
    INFO:chroma:starting view [1024, 576] 
    INFO:chroma:create Camera 
    INFO:chroma:Camera.__init__
    INFO:chroma:Camera.__init__ done
    INFO:chroma:start Camera 
    INFO:chroma:join Camera 
    Process 53332 stopped and restarted: thread 1 received signal: SIGCHLD
    Process 53332 exited with status = 0 (0x00000000) 



supplying an input file, somehow yields a trace
-------------------------------------------------

::

    (chroma_env)delta:test blyth$ chroma-cam @chroma.models.lionsolid -i test.root 
    INFO:chroma:Flattening detector mesh...
    INFO:chroma:  triangles: 74358
    INFO:chroma:  vertices:  37181
    INFO:chroma:Loading BVH "default" for geometry from cache.
    INFO:chroma:loaded geometry <chroma.geometry.Geometry object at 0x1140cc4d0> 
    INFO:chroma:Camera.__init__
    INFO:chroma:Camera.__init__ done
    INFO:chroma:starting viewer [1024, 576] 

     *** Break *** segmentation violation
     Generating stack trace...
     0x00007fff89e60de7 in __CFXNotificationCenterCreate (in CoreFoundation) + 343
     0x00007fff89e60c7a in __CFNotificationCenterGetDistributedCenter_block_invoke (in CoreFoundation) + 26
     0x00007fff86de32ad in _dispatch_client_callout (in libdispatch.dylib) + 8
     0x00007fff86de321c in dispatch_once_f (in libdispatch.dylib) + 79
     0x00007fff89e911a3 in CFNotificationCenterGetDistributedCenter (in CoreFoundation) + 83
     0x00007fff89e910e7 in __71+[CFPrefsSource withSourceForIdentifier:user:byHost:container:perform:]_block_invoke (in CoreFoundation) + 23
     0x00007fff86de32ad in _dispatch_client_callout (in libdispatch.dylib) + 8
     0x00007fff86de321c in dispatch_once_f (in libdispatch.dylib) + 79
     0x00007fff89e90f2a in +[CFPrefsSource withSourceForIdentifier:user:byHost:container:perform:] (in CoreFoundation) + 474
     0x00007fff89e90d3b in -[CFPrefsSearchListSource addSourceForIdentifier:user:byHost:] (in CoreFoundation) + 123
     0x00007fff89e8be43 in +[CFPrefsSearchListSource withSearchListForIdentifier:perform:] (in CoreFoundation) + 323
     0x00007fff89e8bcb8 in CFPreferencesCopyAppValue (in CoreFoundation) + 168
     0x00007fff89ea5293 in ___CFBundleCopyUserLanguages_block_invoke (in CoreFoundation) + 35
     0x00007fff86de32ad in _dispatch_client_callout (in libdispatch.dylib) + 8
     0x00007fff86de321c in dispatch_once_f (in libdispatch.dylib) + 79
     0x00007fff89ea516f in _CFBundleAddPreferredLprojNamesInDirectory (in CoreFoundation) + 911
     0x00007fff89ea3cab in _CFBundleGetLanguageSearchList (in CoreFoundation) + 107
     0x00007fff89ea2cab in _CFBundleCreateQueryTableAtPath (in CoreFoundation) + 539
     0x00007fff89ea2a17 in _CFBundleCopyQueryTable (in CoreFoundation) + 263
     0x00007fff89ea22c3 in _CFBundleCopyURLsOfKey (in CoreFoundation) + 339
     0x00007fff89ea1d05 in _CFBundleCopyFindResources (in CoreFoundation) + 1637
     0x00007fff89ea1693 in CFBundleCopyResourceURL (in CoreFoundation) + 67
     0x00007fff89ea14d1 in CFBundleGetLocalInfoDictionary (in CoreFoundation) + 209
     0x00007fff89ea13c1 in CFBundleGetValueForInfoDictionaryKey (in CoreFoundation) + 33
     0x00007fff8f94b925 in CGSServerPort (in CoreGraphics) + 281
     0x00007fff8f94b7ce in CGSScoreboard (in CoreGraphics) + 23
     0x00007fff8f94b62b in initDisplayState (in CoreGraphics) + 91
     0x00007fff8f94b4d9 in initDisplayMappings (in CoreGraphics) + 29
     0x00007fff8f94accf in __CGSInitialize_block_invoke (in CoreGraphics) + 26
     0x00007fff86de32ad in _dispatch_client_callout (in libdispatch.dylib) + 8
     0x00007fff86de321c in dispatch_once_f (in libdispatch.dylib) + 79
     0x00007fff8f95121d in CGSMainDisplayID (in CoreGraphics) + 20
     0x000000010c9efaf5 in MacOS_WMAvailable (in MacOS.so) + 53
     0x000000010c7d1010 in PyEval_EvalFrameEx (in Python) + 7712
     0x000000010c7cf076 in PyEval_EvalCodeEx (in Python) + 1734
     0x000000010c7620c6 in function_call (in Python) + 342
     0x000000010c73e665 in PyObject_Call (in Python) + 101
     0x000000010c73eb34 in PyObject_CallMethod (in Python) + 388
     0x0000000110683f1d in PyGame_Video_AutoInit (in base.so) + 77
     0x0000000110684529 in init (in base.so) + 185
     0x000000010c7d1fa6 in PyEval_EvalFrameEx (in Python) + 11702
     0x000000010c7d5ed2 in fast_function (in Python) + 194
     0x000000010c7d228b in PyEval_EvalFrameEx (in Python) + 12443
     0x000000010c7d5ed2 in fast_function (in Python) + 194
     0x000000010c7d228b in PyEval_EvalFrameEx (in Python) + 12443
     0x000000010c7cf076 in PyEval_EvalCodeEx (in Python) + 1734
     0x000000010c7620c6 in function_call (in Python) + 342
     0x000000010c73e665 in PyObject_Call (in Python) + 101
     0x000000010c74a7b6 in instancemethod_call (in Python) + 182
     0x000000010c73e665 in PyObject_Call (in Python) + 101
     0x000000010c7959fd in slot_tp_init (in Python) + 141
     0x000000010c78ffe2 in type_call (in Python) + 338
     0x000000010c73e665 in PyObject_Call (in Python) + 101
     0x000000010c7d30b4 in PyEval_EvalFrameEx (in Python) + 16068
     0x000000010c7d5ed2 in fast_function (in Python) + 194
     0x000000010c7d228b in PyEval_EvalFrameEx (in Python) + 12443
     0x000000010c7cf076 in PyEval_EvalCodeEx (in Python) + 1734
     0x000000010c7ce9a6 in PyEval_EvalCode (in Python) + 54
     0x000000010c7f6611 in PyRun_FileExFlags (in Python) + 161
     0x000000010c7cafe6 in builtin_execfile (in Python) + 502
     0x000000010c7d1010 in PyEval_EvalFrameEx (in Python) + 7712
     0x000000010c7cf076 in PyEval_EvalCodeEx (in Python) + 1734
     0x000000010c7ce9a6 in PyEval_EvalCode (in Python) + 54
     0x000000010c7f6611 in PyRun_FileExFlags (in Python) + 161
     0x000000010c7f615e in PyRun_SimpleFileExFlags (in Python) + 718
     0x000000010c80a002 in Py_Main (in Python) + 3314
     0x00007fff90bdc5fd in start (in libdyld.dylib) + 1
    (chroma_env)delta:test blyth$ 




Looking for minimal tickle
----------------------------

Try to tickle this with pygame examples.::

    (chroma_env)delta:test blyth$ python -c "from pygame.macosx import Video_AutoInit ; print '\n'.join(dir(Video_AutoInit))"
    __call__
    __class__
    __closure__
    ...



checking the cache
---------------------


::

    (chroma_env)delta:chroma blyth$ l ~/.chroma/
    bvh/       geo/       root.C     root_C.d   root_C.so  
    (chroma_env)delta:chroma blyth$ l ~/.chroma/geo/
    (chroma_env)delta:chroma blyth$ l ~/.chroma/bvh/
    total 0
    drwxr-xr-x  3 blyth  staff  102 Jan 20 16:27 c61432f54300d20b1b936b38da17a7e6
    drwxr-xr-x  3 blyth  staff  102 Jan 20 15:54 5bb27feefcb2dda31256bf6e755cb451
    drwxr-xr-x  3 blyth  staff  102 Jan 16 20:44 a840c5fc071c8fd80c08ce8b298cc4d0
    drwxr-xr-x  3 blyth  staff  102 Jan 16 20:10 2839c840dbc9bd95a3af0114b07ebc2e
    (chroma_env)delta:chroma blyth$ date
    Mon Jan 20 16:28:20 CST 2014
    (chroma_env)delta:chroma blyth$ 



red herrings
--------------


* https://bitbucket.org/pygame/pygame/src/73cefe45328a/src/base.c

  * the code that segments 

* http://osdir.com/ml/python.pygame/2005-10/msg00166.html
 
  * suggests SDL envvars pointing to drivers ?


* http://stackoverflow.com/questions/18768967/python-segmentation-fault-11-on-osx

  * suggests readline related i




::

    chroma_env)delta:lib-dynload blyth$ pwd
    /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/lib-dynload
    (chroma_env)delta:lib-dynload blyth$ sudo mv readline.so readline.so.disabled
    Password:
    (chroma_env)delta:lib-dynload blyth$ sudo mv readline.so.disabled readline.so
    (chroma_env)delta:lib-dynload blyth$ 




