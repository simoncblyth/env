PyCUDA Debugging
==================


cuda-gdb
--------

::

    (chroma_env)delta:chroma_camera blyth$ cuda-gdb --args python -m pycuda.debug simplecamera.py -s3199 -d3 -f10 --eye=0,1,0 --lookat=10,0,10  -i
    NVIDIA (R) CUDA Debugger
    5.5 release
    Portions Copyright (C) 2007-2013 NVIDIA Corporation
    GNU gdb (GDB) 7.2
    Copyright (C) 2010 Free Software Foundation, Inc.
    License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
    This is free software: you are free to change and redistribute it.
    There is NO WARRANTY, to the extent permitted by law.  Type "show copying"
    and "show warranty" for details.
    This GDB was configured as "x86_64-apple-darwin11.4.2".
    For bug reporting instructions, please see:
    <http://www.gnu.org/software/gdb/bugs/>...
    Reading symbols from /usr/local/env/chroma_env/bin/python...(no debugging symbols found)...done.
    (cuda-gdb) r
    Starting program: /usr/local/env/chroma_env/bin/python -m pycuda.debug simplecamera.py -s3199 -d3 -f10 --eye=0,1,0 --lookat=10,0,10 -i
    warning: Could not open OSO archive file "/opt/local/var/macports/build/_opt_mports_dports_lang_python27/python27/work/Python-2.7.6/libpython2.7.a"
    warning: Could not open OSO archive file "/BinaryCache/Libsyscall/Libsyscall-2422.90.20~2/Symbols/BuiltProducts/libsystem_kernel.a"
    warning: `/private/var/tmp/Libsyscall/Libsyscall-2422.90.20~2/Libsyscall.build/Libsyscall_dynamic.build/Objects-normal/x86_64/_libc_funcptr.o': can't open to read symbols: No such file or directory.
    warning: `/private/var/tmp/Libsyscall/Libsyscall-2422.90.20~2/Libsyscall.build/Libsyscall_dynamic.build/Objects-normal/x86_64/kernel_vers.o': can't open to read symbols: No such file or directory.
    warning: `/private/var/tmp/Libsyscall/Libsyscall-2422.90.20~2/Libsyscall.build/Libsyscall_dynamic.build/Objects-normal/x86_64/memcpy.o': can't open to read symbols: No such file or directory.
    warning: `/private/var/tmp/Libsyscall/Libsyscall-2422.90.20~2/Libsyscall.build/Libsyscall_dynamic.build/Objects-normal/x86_64/strcmp.o': can't open to read symbols: No such file or directory.
    ...
    WARNING:env.geant4.geometry.collada.daenode:failed to find parent for   top.0             -  (failure expected only for root node)
    WARNING:env.geant4.geometry.collada.collada_to_chroma:setting parent_material to __dd__Materials__Vacuum0xaf1d298 as parent is None for node top.0 
    INFO:chroma:Flattening detector mesh...
    INFO:chroma:  triangles: 2483650
    INFO:chroma:  vertices:  1264049
    INFO:chroma:Loading BVH "default" for geometry from cache.
    INFO:chroma:Camera.__init__
    -s/--solid           : 3199 
    -e/--eye             : [ 0.  1.  0.] 
    -a/--lookat          : [ 10.   0.  10.] 
    -u/--up              : [-1.  0.  0.] 
    -r/--size            : [640, 480] 
    -d/--alpha-max       : 3 
    -f/--focal-length    : 10 
    -r/--radius          : 0.5 
    [New Thread 0x1653 of process 960]
    warning: `/opt/local/var/macports/build/_opt_mports_dports_lang_python27/python27/work/Python-2.7.6/build/temp.macosx-10.9-x86_64-2.7/opt/local/var/macports/build/_opt_mports_dports_lang_python27/python27/work/Python-2.7.6/Mac/Modules/MacOS.o': can't open to read symbols: No such file or directory.
    warning: `/opt/local/var/macports/build/_opt_mports_dports_lang_python27/python27/work/Python-2.7.6/build/temp.macosx-10.9-x86_64-2.7/opt/local/var/macports/build/_opt_mports_dports_lang_python27/python27/work/Python-2.7.6/Modules/parsermodule.o': can't open to read symbols: No such file or directory.
    warning: `/opt/local/var/macports/build/_opt_mports_dports_python_py-game/py27-game/work/pygame-1.9.1release/build/temp.macosx-10.9-x86_64-2.7/src/sdlmain_osx.o': can't open to read symbols: No such file or directory.
    [New Thread 0x181b of process 960]
    [New Thread 0x1903 of process 960]
    [New Thread 0x1a03 of process 960]
    [New Thread 0x1b03 of process 960]
    warning: `/private/var/tmp/CoreServicesInternal/CoreServicesInternal-184.9~1/CoreServicesInternal.build/CoreServicesInternal.build/Objects-normal/x86_64/BookmarkCommon.o': can't open to read symbols: No such file or directory.
    warning: `/private/var/tmp/CoreServicesInternal/CoreServicesInternal-184.9~1/CoreServicesInternal.build/CoreServicesInternal.build/Objects-normal/x86_64/BookmarkData.o': can't open to read symbols: No such file or directory.
    ...
    ...
    warning: `/private/var/tmp/CoreServicesInternal/CoreServicesInternal-184.9~1/CoreServicesInternal.build/CoreServicesInternal.build/Objects-normal/x86_64/ScopedBookmarksClient.o': can't open to read symbols: No such file or directory.
    warning: `/private/var/tmp/CoreServicesInternal/CoreServicesInternal-184.9~1/CoreServicesInternal.build/CoreServicesInternal.build/Objects-normal/x86_64/ftsattr.o': can't open to read symbols: No such file or directory.
    [New Thread 0x1c0b of process 960]
    [New Thread 0x1d23 of process 960]
    [New Thread 0x1e1b of process 960]
    INFO:chroma:using bounds (array([ -16701.390625  , -801625.6875    ,   -8942.78808594], dtype=float32), array([ -16431.1953125 , -801339.5625    ,   -8742.21191406], dtype=float32)) for solid_index 3199 
    cam_up (axis2)   [-0.50248756 -0.04975124  0.49751244] 
    left   (axis1)   [ 0.          0.70534562  0.07053456] 
    forward          [ 0.70534562 -0.07053456  0.70534562] 
    radius               : 0.5 
    ball_center          : [ 320.  240.] 
    ball_radius          : 200.0 
    [New Thread 0x1f2b of process 960]
    [New Thread 0x15d7 of process 960]
    [Context Create of context 0x113aa6e00 on Device 0]
    A device about to be used for compute may already be in use for graphics.
    This is an unsupported scenario. Further debugging might be unsafe. Aborting.
    Disable the 'cuda gpu_busy_check' option to bypass the checking mechanism.
    (cuda-gdb) 




* http://stackoverflow.com/questions/19138670/cuda-debug-in-mac-error-a-device-about-to-be-used-for-compute-may-already-be


* :google:`A device about to be used for compute may already be in use for graphics.`

* https://devtalk.nvidia.com/default/topic/627584/cuda-gdb/cuda-gdb-on-mavericks/

Debugging with the window server running (desktop) on Mac is only supported
when the device is not used for graphics. This is what they debugger `cuda gpu_busy_check` 
verifies. You can override it at your own risks. If the device
is used for display, hitting a breakpoint in your cuda application will freeze
the display.

To help mitigate the problem, you can try to:

#. disable any form of hardware acceleration
#. do not use anything openGL and the like
#. disable Automatic Graphics Switching in the System Settings->Energy Saver tab.


Some more information can be found at 

* http://docs.nvidia.com/cuda/cuda-gdb/index.html#single-gpu-debugging



OSX console login
--------------------

Need to switch off auto login and add name and password in `Prefs > Users and Groups`

* http://www.tuaw.com/2011/02/26/little-known-options-for-the-os-x-login-window/


* http://apple.stackexchange.com/questions/85742/console-does-not-work

Enabled root user, but still cannot do the console login. The trick is
to login as `>console` with the greater than sign, then you enter an
admin password.  The result is tiny text on retina screen, notes that 
network and SVN works in this mode.

Need to work out how to tickle the issue in headless manner.


pygame headless
---------------

See `/opt/local/share/doc/py27-game/examples/headless_no_windows_needed.py`

*


observations
-------------

Device memory usage logged varies a lot::

    INFO:chroma:Optimization: Sufficient memory to move triangles onto GPU
    INFO:chroma:Optimization: Sufficient memory to move vertices onto GPU
    INFO:chroma:device usage:
    ----------
    nodes             2.8M  44.7M
    total                   44.7M
    ----------
    device total             2.1G
    device used              1.6G
    device free            568.2M



 


