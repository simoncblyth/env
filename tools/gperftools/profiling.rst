Profiling Mechanics
=====================

.. contents:: :local:

Example
-----------------

For usage example see :e:`muon_simulation/profiling/base/analysis/`


Process Memory Maps
--------------------

* http://www.trilithium.com/johan/2005/08/linux-gate/
* http://stackoverflow.com/questions/1401359/understanding-linux-proc-id-maps

Each row in /proc/$PID/maps describes a region of contiguous virtual memory in a process or thread. 

* http://google-perftools.googlecode.com/svn/trunk/doc/cpuprofile-fileformat.html
* :google:`/proc/pid/maps`
* http://linux.die.net/man/5/proc 

::

    /proc/[pid]/maps
    A file containing the currently mapped memory regions and their access permissions.
    The format is:

    address           perms offset  dev   inode   pathname
    08048000-08056000 r-xp 00000000 03:0c 64593   /usr/sbin/gpm
    08056000-08058000 rw-p 0000d000 03:0c 64593   /usr/sbin/gpm
    08058000-0805b000 rwxp 00000000 00:00 0
    40000000-40013000 r-xp 00000000 03:0c 4165    /lib/ld-2.2.4.so
    40013000-40015000 rw-p 00012000 03:0c 4165    /lib/ld-2.2.4.so
    4001f000-40135000 r-xp 00000000 03:0c 45494   /lib/libc-2.2.4.so
    40135000-4013e000 rw-p 00115000 03:0c 45494   /lib/libc-2.2.4.so
    4013e000-40142000 rw-p 00000000 00:00 0
    bffff000-c0000000 rwxp 00000000 00:00 0

    where "address" is the address space in the process that it occupies, "perms" is a set of permissions:
    r = read
    w = write
    x = execute
    s = shared
    p = private (copy on write)

    "offset" is the offset into the file/whatever, 
    "dev" is the device (major:minor), and 
    "inode" is the inode on that device. 0 indicates that no inode is associated with the memory region, as the case would be with BSS (uninitialized data).


Sample Collection
------------------

::

    [blyth@belle7 20130816-1754]$ LD_PRELOAD=/usr/local/lib/libprofiler.so CPUPROFILE=/tmp/nuwa.perf $(which python) opw-sim.py 

::

    [blyth@belle7 e]$ ll /tmp/nuwa.perf*
    -rw-rw-r-- 1 blyth blyth    2473 Aug 19 15:30 /tmp/nuwa.perf_4090
    -rw-rw-r-- 1 blyth blyth    2473 Aug 19 15:30 /tmp/nuwa.perf_4088
    -rw-rw-r-- 1 blyth blyth    2305 Aug 19 15:30 /tmp/nuwa.perf_4086
    -rw-rw-r-- 1 blyth blyth 7965722 Aug 19 15:40 /tmp/nuwa.perf

The small ones look like PID maps only `/proc/PID/maps` without sample info (shortlived forks ?), for the /bin/bash and /usr/bin/hostid processes that got forked.
The large one has a binary head and a big ascii tail with a large library map.



pprof sample analysis
-----------------------

::

    [blyth@belle7 20130816-1754]$ which pprof
    /usr/local/bin/pprof



Text Profile 
~~~~~~~~~~~~~

::

    [blyth@belle7 20130816-1754]$ pprof --text $(which python) /tmp/nuwa.perf > /tmp/nuwa.perf.txt
    Using local file /data1/env/local/dyb/external/Python/2.7/i686-slc5-gcc41-dbg/bin/python.
    Using local file /tmp/nuwa.perf.
    Removing _init from all stack traces.


Source line hot spots 
~~~~~~~~~~~~~~~~~~~~~~~~


graphical 
~~~~~~~~~

::

    [blyth@belle7 ~]$ pprof --pdf $(which python) /tmp/nuwa.perf > /tmp/nuwa.perf.pdf
    Using local file /data1/env/local/dyb/external/Python/2.7/i686-slc5-gcc41-dbg/bin/python.
    Using local file /tmp/nuwa.perf.
    Removing _init from all stack traces.
    Dropping nodes with <= 255 samples; edges with <= 51 abs(samples)



kcachegrind
~~~~~~~~~~~~~~

Presentation of the profile data



* http://kcachegrind.sourceforge.net/html/Home.html




