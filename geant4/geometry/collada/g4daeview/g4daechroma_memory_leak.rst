G4DAEChroma Memory Leak
=========================

Executive Summary
-------------------

Initial concerns over large VSIZE appear unfounded, just the 
way OSX does its totting up.  
Question remains: why is CPU usage so high ? 
The GPU should be doing the work, CPU just 
doing copies and marshalling ?



Initial Observation of 30G VSIZE
-----------------------------------

Running a mocknuwa scan matrix of 16(configs)x48(batches) 
reveals the python g4daechroma.py 
propagating process to balloon in VSIZE shown by top.
Got to 30G and the machine became sluggish before interrupting
it at config 11.

::

    mocknuwa-scan(){
       mocknuwa-runenv 
       MockNuWa 1:49 1:17
    }


Large VSIZE
-------------

* http://manytricks.com/blog/?p=2471

Not ballooning to 30G
-----------------------

VSIZE is steady at 30G, and starts there even before 
doing any propagations.

Doing an activity monitor sample gives a hint of how it gets to 30G.  
All the dependencies of chroma, even although they are not 
being used are presumably contributing to the 30G.

* pygame
* opengl
* geant4 

* avoiding kitchen sink `__init__.py` gets to 30.75 G 
  by avoiding unused dependencies like pygame and geant4



vmmap
--------

28G listed under VM_ALLOCATE 


::

    (chroma_env)delta:~ blyth$ pgrep python
    45624
    (chroma_env)delta:~ blyth$ vmmap 45624
    Virtual Memory Map of process 45624 (python)
    Output report format:  2.2  -- 64-bit process

    ==== Non-writable regions for process 45624
    __TEXT                 0000000105b60000-0000000105b62000 [    8K] r-x/rwx SM=COW  /usr/local/env/chroma_env/bin/python
    __LINKEDIT             0000000105b63000-0000000105b64000 [    4K] r--/rwx SM=COW  /usr/local/env/chroma_env/bin/python

    ...

    __LINKEDIT             0000000117408000-000000011740b000 [   12K] r--/rwx SM=COW  /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/_pvt_struct.so
    STACK GUARD            000000011f216000-000000011f217000 [    4K] ---/rwx SM=NUL  stack guard for thread 2
    STACK GUARD            000000011f299000-000000011f29a000 [    4K] ---/rwx SM=NUL  stack guard for thread 3
    VM_ALLOCATE            0000000200000000-0000000900000000 [ 28.0G] ---/rwx SM=NUL  
    STACK GUARD            00007fff560a0000-00007fff598a0000 [ 56.0M] ---/rwx SM=NUL  stack guard for thread 0
    __TEXT                 00007fff65ccf000-00007fff65d03000 [  208K] r-x/rwx SM=COW  /usr/lib/dyld
    __LINKEDIT             00007fff65d42000-00007fff65d56000 [   80K] r--/rwx SM=COW  /usr/lib/dyld
    __TEXT                 00007fff87ef7000-00007fff87f09000 [   72K] r-x/r-x SM=COW  /usr/lib/libz.1.2.5.dylib
    ...

    ==== Summary for process 45624
    ReadOnly portion of Libraries: Total=146.2M resident=59.2M(40%) swapped_out_or_unallocated=87.1M(60%)
    Writable regions: Total=404.6M written=108.9M(27%) resident=361.6M(89%) swapped_out=0K(0%) unallocated=43.0M(11%)

    REGION TYPE                      VIRTUAL
    ===========                      =======
    IOKit                             109.2M
    IOKit (reserved)                      8K        reserved VM address space (unallocated)
    Kernel Alloc Once                     4K
    MALLOC                            194.8M        see MALLOC ZONE table below
    MALLOC (admin)                       44K
    MALLOC freed, no zone              13.9M
    MALLOC_LARGE                       70.7M
    STACK GUARD                        56.0M
    Stack                              9752K
    VM_ALLOCATE                        28.0G
    __DATA                             31.3M
    __LINKEDIT                         71.7M
    __NV_CUDA                          6504K
    __TEXT                             74.5M
    __UNICODE                           544K
    shared memory                         4K
    ===========                      =======
    TOTAL                              28.6G
    TOTAL, minus reserved VM space     28.6G



vmmap -resident
-----------------

::

    (chroma_env)delta:~ blyth$ vmmap -resident 45624
    Virtual Memory Map of process 45624 (python)
    Output report format:  2.2  -- 64-bit process

    ==== Non-writable regions for process 45624
    __TEXT                 0000000105b60000-0000000105b62000 [    8K     8K] r-x/rwx SM=COW  /usr/local/env/chroma_env/bin/python
    __LINKEDIT             0000000105b63000-0000000105b64000 [    4K     4K] r--/rwx SM=COW  /usr/local/env/chroma_env/bin/python

    ...
    __TEXT                 0000000117402000-0000000117406000 [   16K    16K] r-x/rwx SM=COW  /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/_pvt_struct.so
    __LINKEDIT             0000000117408000-000000011740b000 [   12K    12K] r--/rwx SM=COW  /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/_pvt_struct.so
    STACK GUARD            000000011f216000-000000011f217000 [    4K     0K] ---/rwx SM=NUL  stack guard for thread 2
    STACK GUARD            000000011f299000-000000011f29a000 [    4K     0K] ---/rwx SM=NUL  stack guard for thread 3
    VM_ALLOCATE            0000000200000000-0000000900000000 [ 28.0G     0K] ---/rwx SM=NUL  
    STACK GUARD            00007fff560a0000-00007fff598a0000 [ 56.0M     0K] ---/rwx SM=NUL  stack guard for thread 0
    __TEXT                 00007fff65ccf000-00007fff65d03000 [  208K   208K] r-x/rwx SM=COW  /usr/lib/dyld
    __LINKEDIT             00007fff65d42000-00007fff65d56000 [   80K    80K] r--/rwx SM=COW  /usr/lib/dyld


    ==== Legend
    SM=sharing mode:  
        COW=copy_on_write PRV=private NUL=empty ALI=aliased 
        SHM=shared ZER=zero_filled S/A=shared_alias

    ==== Summary for process 45624
    ReadOnly portion of Libraries: Total=146.2M resident=59.2M(40%) swapped_out_or_unallocated=87.1M(60%)
    Writable regions: Total=414.6M written=108.9M(26%) resident=361.7M(87%) swapped_out=0K(0%) unallocated=53.0M(13%)

    REGION TYPE                      VIRTUAL RESIDENT
    ===========                      ======= ========
    IOKit                             109.2M   109.2M
    IOKit (reserved)                      8K       0K        reserved VM address space (unallocated)
    Kernel Alloc Once                     4K       4K
    MALLOC                            204.8M   166.6M        see MALLOC ZONE table below
    MALLOC (admin)                       44K      12K
    MALLOC freed, no zone              13.9M    13.9M
    MALLOC_LARGE                       70.7M    70.7M
    STACK GUARD                        56.0M       0K
    Stack                              9752K     112K
    VM_ALLOCATE                        28.0G      12K
    __DATA                             31.3M    6116K
    __LINKEDIT                         71.7M    18.3M
    __NV_CUDA                          6504K       8K
    __TEXT                             74.5M    40.9M
    __UNICODE                           544K     544K
    shared memory                         4K       4K
    ===========                      ======= ========
    TOTAL                              28.6G   426.2M
    TOTAL, minus reserved VM space     28.6G   426.2M


                                     VIRTUAL   RESIDENT ALLOCATION      BYTES
    MALLOC ZONE                         SIZE       SIZE      COUNT  ALLOCATED  % FULL
    ===========                      =======  =========  =========  =========  ======
    DefaultMallocZone_0x105b64000     204.4M     166.5M      23534     166.1M     81%
    GFXMallocZone_0x1093a9000             0K         0K          0         0K
    ===========                      =======  =========  =========  =========  ======
    TOTAL                             204.4M     166.5M      23534     166.1M     81%




