rtow-source(){   echo ${BASH_SOURCE} ; }
rtow-edir(){ echo $(dirname $(rtow-source)) ; }
rtow-ecd(){  cd $(rtow-edir); }
rtow-dir(){  echo $LOCAL_BASE/env/graphics/RTOW-OptiX/rtow ; }
rtow-cd(){   cd $(rtow-dir); }
rtow-vi(){   vi $(rtow-source) ; }
rtow-env(){  elocal- ; }
rtow-usage(){ cat << EOU

rtow : RTOW-OptiX (Ingo Wald's OptiX version of Pete Shirleys 'Ray Tracing in One Weekend' final chapter example
===================================================================================================================

* https://ingowald.blog/
* https://github.com/ingowald/RTOW-OptiX
* https://github.com/simoncblyth/RTOW-OptiX

Objective
----------

Want to be able to see the effect of RTX execution model on 
ray tracing performance : so need something to stress the GPU.
Also interesting to see approaches to OptiX performance
measurements.


finalChapter_iterative with default resolution 1200x800
-----------------------------------------------------------

OptiX_600
~~~~~~~~~~~~~~

::

    [blyth@localhost rtow.build]$ ldd ./finalChapter_iterative
        linux-vdso.so.1 =>  (0x00007fff700f3000)
        liboptix.so.6.0.0 => /usr/local/OptiX_600/lib64/liboptix.so.6.0.0 (0x00007f19b869e000)
        libstdc++.so.6 => /lib64/libstdc++.so.6 (0x00007f19b8397000)
        libm.so.6 => /lib64/libm.so.6 (0x00007f19b8095000)
        libgcc_s.so.1 => /lib64/libgcc_s.so.1 (0x00007f19b7e7f000)
        libc.so.6 => /lib64/libc.so.6 (0x00007f19b7ab2000)
        libdl.so.2 => /lib64/libdl.so.2 (0x00007f19b78ae000)
        /lib64/ld-linux-x86-64.so.2 (0x00007f19b896d000)



finalChapter_iterative summary 
--------------------------------------------------------


======================   =======   ====================  =========  =============================================
CUDA_VISIBLE_DEVICES       RTX       avg of 10 (1)        one (2)            notes
======================   =======   ====================  =========  =============================================
    1:TITAN RTX            ASIS       0.4315               5.874            
    1:TITAN RTX            ON         0.4323
    0:TITAN V              ASIS       0.6669               9.131       9.131/5.874=1.554  0.6669/0.4315 = 1.5455
    0:TITAN V              ON         0.6677
    0:TITAN V              OFF        0.6781
    1:TITAN RTX            OFF        0.8048                           0.8048/0.4319 = 1.8633
======================   =======   ====================  =========  =============================================


* RTX disabled : the TITAN V is faster (0.66 vs 0.80)  
* RTX enabled : the TITAN RTX is faster (0.43 vs 0.66) 

These times are similar to those from the author https://ingowald.blog/2018/11/21/rtow-in-optix-fun-with-curand/

* (1) default resolution 1200,800  = 960,000              ~1M
* (2) double screen resolution 5120,2880 = 14,745,600    ~15M



To see significant differnce between TITAN V and TITAN RTX have to increase resolution substantially
-------------------------------------------------------------------------------------------------------

::

    [blyth@localhost rtow.build]$ CUDA_VISIBLE_DEVICES=0 ./finalChapter_iterative --size 5120,2880
    ./finalChapter_iterative rtx -1 stack 3000 width 5120 height 2880 rtx_default -1 stack_default 3000 size_default 1200,800
    done building optix data structures, which took 0.1404 seconds
    done rendering, which took 9.131 seconds (for 128 paths per pixel)

    [blyth@localhost rtow.build]$ CUDA_VISIBLE_DEVICES=1 ./finalChapter_iterative --size 5120,2880
    ./finalChapter_iterative rtx -1 stack 3000 width 5120 height 2880 rtx_default -1 stack_default 3000 size_default 1200,800
    done building optix data structures, which took 0.1334 seconds
    done rendering, which took 5.874 seconds (for 128 paths per pixel)







TITAN V : 0.67-0.68s
~~~~~~~~~~~~~~~~~~~~~~

* varying stack size seems to have no effect : is it being ignored ?
* switching RTX ASIS/OFF/ON  has no significant effect (slight suspicion that switching it OFF increases time)

::

    blyth@localhost rtow.build]$ CUDA_VISIBLE_DEVICES=0 ./finalChapter_iterative --repeat 10 --rtx -1
    ./finalChapter_iterative rtx -1 stack 3000 width 1200 height 800 repeat 10 rtx_default -1 stack_default 3000 size_default 1200,800 repeat_default 5
    done building optix data structures, which took 0.125 seconds
    done rendering, which took 0.6673 seconds (for 128 paths per pixel) r 0
    done rendering, which took 0.6666 seconds (for 128 paths per pixel) r 1
    done rendering, which took 0.6613 seconds (for 128 paths per pixel) r 2
    done rendering, which took 0.6665 seconds (for 128 paths per pixel) r 3
    done rendering, which took 0.6647 seconds (for 128 paths per pixel) r 4
    done rendering, which took 0.6714 seconds (for 128 paths per pixel) r 5
    done rendering, which took 0.6767 seconds (for 128 paths per pixel) r 6
    done rendering, which took 0.6638 seconds (for 128 paths per pixel) r 7
    done rendering, which took 0.6635 seconds (for 128 paths per pixel) r 8
    done rendering, which took 0.6676 seconds (for 128 paths per pixel) r 9
                           avg 0.6669
    [blyth@localhost rtow.build]$ CUDA_VISIBLE_DEVICES=0 ./finalChapter_iterative --repeat 10 --rtx 0
    ./finalChapter_iterative rtx 0 stack 3000 width 1200 height 800 repeat 10 rtx_default -1 stack_default 3000 size_default 1200,800 repeat_default 5
    done building optix data structures, which took 0.6168 seconds
    done rendering, which took 0.6828 seconds (for 128 paths per pixel) r 0
    done rendering, which took 0.6792 seconds (for 128 paths per pixel) r 1
    done rendering, which took 0.6729 seconds (for 128 paths per pixel) r 2
    done rendering, which took 0.692 seconds (for 128 paths per pixel) r 3
    done rendering, which took 0.6769 seconds (for 128 paths per pixel) r 4
    done rendering, which took 0.6726 seconds (for 128 paths per pixel) r 5
    done rendering, which took 0.6712 seconds (for 128 paths per pixel) r 6
    done rendering, which took 0.6821 seconds (for 128 paths per pixel) r 7
    done rendering, which took 0.6734 seconds (for 128 paths per pixel) r 8
    done rendering, which took 0.6781 seconds (for 128 paths per pixel) r 9
                           avg 0.6781
    [blyth@localhost rtow.build]$ CUDA_VISIBLE_DEVICES=0 ./finalChapter_iterative --repeat 10 --rtx 1
    ./finalChapter_iterative rtx 1 stack 3000 width 1200 height 800 repeat 10 rtx_default -1 stack_default 3000 size_default 1200,800 repeat_default 5
    done building optix data structures, which took 0.1345 seconds
    done rendering, which took 0.6669 seconds (for 128 paths per pixel) r 0
    done rendering, which took 0.6667 seconds (for 128 paths per pixel) r 1
    done rendering, which took 0.6699 seconds (for 128 paths per pixel) r 2
    done rendering, which took 0.6648 seconds (for 128 paths per pixel) r 3
    done rendering, which took 0.6734 seconds (for 128 paths per pixel) r 4
    done rendering, which took 0.6696 seconds (for 128 paths per pixel) r 5
    done rendering, which took 0.6652 seconds (for 128 paths per pixel) r 6
    done rendering, which took 0.6655 seconds (for 128 paths per pixel) r 7
    done rendering, which took 0.6704 seconds (for 128 paths per pixel) r 8
    done rendering, which took 0.6646 seconds (for 128 paths per pixel) r 9
                           avg 0.6677



TITAN RTX : contrary to the OptiX_600 documentation RTX is enabled by default : switching it off increases time by factor of almost 2 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:: 

    [blyth@localhost rtow.build]$ CUDA_VISIBLE_DEVICES=1 ./finalChapter_iterative --repeat 10 --rtx -1
    ./finalChapter_iterative rtx -1 stack 3000 width 1200 height 800 repeat 10 rtx_default -1 stack_default 3000 size_default 1200,800 repeat_default 5
    done building optix data structures, which took 0.1303 seconds
    done rendering, which took 0.4643 seconds (for 128 paths per pixel) r 0
    done rendering, which took 0.4241 seconds (for 128 paths per pixel) r 1
    done rendering, which took 0.4254 seconds (for 128 paths per pixel) r 2
    done rendering, which took 0.4277 seconds (for 128 paths per pixel) r 3
    done rendering, which took 0.4322 seconds (for 128 paths per pixel) r 4
    done rendering, which took 0.4291 seconds (for 128 paths per pixel) r 5
    done rendering, which took 0.4276 seconds (for 128 paths per pixel) r 6
    done rendering, which took 0.4279 seconds (for 128 paths per pixel) r 7
    done rendering, which took 0.4292 seconds (for 128 paths per pixel) r 8
    done rendering, which took 0.4276 seconds (for 128 paths per pixel) r 9
                           avg 0.4315
    [blyth@localhost rtow.build]$ 
    [blyth@localhost rtow.build]$ CUDA_VISIBLE_DEVICES=1 ./finalChapter_iterative --repeat 10 --rtx 1
    ./finalChapter_iterative rtx 1 stack 3000 width 1200 height 800 repeat 10 rtx_default -1 stack_default 3000 size_default 1200,800 repeat_default 5
    done building optix data structures, which took 0.1184 seconds
    done rendering, which took 0.4642 seconds (for 128 paths per pixel) r 0
    done rendering, which took 0.4305 seconds (for 128 paths per pixel) r 1
    done rendering, which took 0.4282 seconds (for 128 paths per pixel) r 2
    done rendering, which took 0.4295 seconds (for 128 paths per pixel) r 3
    done rendering, which took 0.4322 seconds (for 128 paths per pixel) r 4
    done rendering, which took 0.4282 seconds (for 128 paths per pixel) r 5
    done rendering, which took 0.4271 seconds (for 128 paths per pixel) r 6
    done rendering, which took 0.4283 seconds (for 128 paths per pixel) r 7
    done rendering, which took 0.4264 seconds (for 128 paths per pixel) r 8
    done rendering, which took 0.4282 seconds (for 128 paths per pixel) r 9
                           avg 0.4323
    [blyth@localhost rtow.build]$ CUDA_VISIBLE_DEVICES=1 ./finalChapter_iterative --repeat 10 --rtx 0
    ./finalChapter_iterative rtx 0 stack 3000 width 1200 height 800 repeat 10 rtx_default -1 stack_default 3000 size_default 1200,800 repeat_default 5
    done building optix data structures, which took 0.5842 seconds
    done rendering, which took 0.8145 seconds (for 128 paths per pixel) r 0
    done rendering, which took 0.8048 seconds (for 128 paths per pixel) r 1
    done rendering, which took 0.7998 seconds (for 128 paths per pixel) r 2
    done rendering, which took 0.8048 seconds (for 128 paths per pixel) r 3
    done rendering, which took 0.8057 seconds (for 128 paths per pixel) r 4
    done rendering, which took 0.8024 seconds (for 128 paths per pixel) r 5
    done rendering, which took 0.8006 seconds (for 128 paths per pixel) r 6
    done rendering, which took 0.8068 seconds (for 128 paths per pixel) r 7
    done rendering, which took 0.8058 seconds (for 128 paths per pixel) r 8
    done rendering, which took 0.8026 seconds (for 128 paths per pixel) r 9
                           avg 0.8048
    [blyth@localhost rtow.build]$ 





TITAN V and TITAN RTX : 0.46-0.49s (same as TITAN RTX)
-------------------------------------------------------

::

    [blyth@localhost rtow.build]$ CUDA_VISIBLE_DEVICES=1,0 ./finalChapter_iterative
    done building optix data structures, which took 0.117 seconds
    done rendering, which took 0.4926 seconds (for 128 paths per pixel)

    [blyth@localhost rtow.build]$ CUDA_VISIBLE_DEVICES=0,1 ./finalChapter_iterative
    done building optix data structures, which took 0.114 seconds
    done rendering, which took 0.4873 seconds (for 128 paths per pixel)

    [blyth@localhost rtow.build]$ CUDA_VISIBLE_DEVICES=1,0 ./finalChapter_iterative
    done building optix data structures, which took 0.1189 seconds
    done rendering, which took 0.4955 seconds (for 128 paths per pixel)

    [blyth@localhost rtow.build]$ CUDA_VISIBLE_DEVICES=1,0 ./finalChapter_iterative
    done building optix data structures, which took 0.1343 seconds
    done rendering, which took 0.4649 seconds (for 128 paths per pixel)



finalChapter_recursive default resolution
--------------------------------------------

::


    blyth@localhost rtow.build]$ CUDA_VISIBLE_DEVICES=0 ./finalChapter_recursive 
    done building optix data structures, which took 0.8102 seconds
    terminate called after throwing an instance of 'optix::Exception'
      what():  Unknown error (Details: Function "RTresult _rtContextLaunch2D(RTcontext, unsigned int, RTsize, RTsize)" caught exception: Encountered a CUDA error: cudaDriver().CuEventSynchronize( m_event ) returned (700): Illegal address)
    Aborted (core dumped)
    [blyth@localhost rtow.build]$ CUDA_VISIBLE_DEVICES=0 ./finalChapter_recursive 
    done building optix data structures, which took 0.1526 seconds
    terminate called after throwing an instance of 'optix::Exception'
      what():  Unknown error (Details: Function "RTresult _rtContextLaunch2D(RTcontext, unsigned int, RTsize, RTsize)" caught exception: Encountered a CUDA error: cudaDriver().CuEventSynchronize( m_event ) returned (700): Illegal address)
    Aborted (core dumped)
    [blyth@localhost rtow.build]$ CUDA_VISIBLE_DEVICES=1 ./finalChapter_recursive 
    done building optix data structures, which took 0.8633 seconds
    done rendering, which took 0.4017 seconds (for 128 paths per pixel)
    [blyth@localhost rtow.build]$ CUDA_VISIBLE_DEVICES=1 ./finalChapter_recursive 
    done building optix data structures, which took 0.1417 seconds
    done rendering, which took 0.4051 seconds (for 128 paths per pixel)
    [blyth@localhost rtow.build]$ CUDA_VISIBLE_DEVICES=1 ./finalChapter_recursive 
    done building optix data structures, which took 0.1161 seconds
    done rendering, which took 0.4059 seconds (for 128 paths per pixel)
    [blyth@localhost rtow.build]$ 


    [blyth@localhost rtow.build]$ CUDA_VISIBLE_DEVICES=1,0 ./finalChapter_recursive 
    done building optix data structures, which took 0.1327 seconds
    done rendering, which took 0.4056 seconds (for 128 paths per pixel)
    [blyth@localhost rtow.build]$ CUDA_VISIBLE_DEVICES=1,0 ./finalChapter_recursive 
    done building optix data structures, which took 0.1205 seconds
    done rendering, which took 0.4077 seconds (for 128 paths per pixel)
    [blyth@localhost rtow.build]$ CUDA_VISIBLE_DEVICES=0,1 ./finalChapter_recursive 
    done building optix data structures, which took 0.12 seconds
    done rendering, which took 0.4018 seconds (for 128 paths per pixel)
    [blyth@localhost rtow.build]$ CUDA_VISIBLE_DEVICES=0,1 ./finalChapter_recursive 
    done building optix data structures, which took 0.123 seconds
    done rendering, which took 0.4085 seconds (for 128 paths per pixel)
    [blyth@localhost rtow.build]$ 
    [blyth@localhost rtow.build]$ CUDA_VISIBLE_DEVICES=0 ./finalChapter_recursive 
    done building optix data structures, which took 0.1459 seconds
    terminate called after throwing an instance of 'optix::Exception'
      what():  Unknown error (Details: Function "RTresult _rtContextLaunch2D(RTcontext, unsigned int, RTsize, RTsize)" caught exception: Encountered a CUDA error: cudaDriver().CuEventSynchronize( m_event ) returned (700): Illegal address)
    Aborted (core dumped)
    [blyth@localhost rtow.build]$ CUDA_VISIBLE_DEVICES=1 ./finalChapter_recursive 
    done building optix data structures, which took 0.1204 seconds
    done rendering, which took 0.4032 seconds (for 128 paths per pixel)
    [blyth@localhost rtow.build]$ 




C++11 needed
----------------

::

    [ 32%] Building CXX object FinalChapter_recursive/CMakeFiles/finalChapter_recursive.dir/finalChapter.cpp.o
    In file included from /usr/include/c++/4.8.2/random:35:0,
                     from /home/blyth/local/env/graphics/RTOW-OptiX/rtow/FinalChapter_recursive/finalChapter.cpp:27:
    /usr/include/c++/4.8.2/bits/c++0x_warning.h:32:2: error: #error This file requires compiler and library support for the ISO C++ 2011 standard. This support is currently experimental, and must be enabled with the -std=c++11 or -std=gnu++11 compiler options.
     #error This file requires compiler and library support for the \
      ^



EOU
}
rtow-get(){
   local dir=$(dirname $(rtow-dir)) &&  mkdir -p $dir && cd $dir

   [ ! -d rtow ] &&  git clone git@github.com:simoncblyth/RTOW-OptiX.git rtow

}


rtow-bdir(){ echo $(rtow-dir).build ; }
rtow-bcd(){  cd $(rtow-bdir) ; }

rtow-cmake()
{
    local sdir=$(rtow-dir) 
    local bdir=$(rtow-bdir) 

    local iwd=$PWD
    mkdir -p $bdir && cd $bdir

    optix-
    cmake $sdir \
           -DOptiX_INSTALL_DIR=$(optix-install-dir)

    cd $iwd
}

rtow-make()
{
    local iwd=$PWD
    rtow-bcd
    make $*
    cd $iwd
}

rtow--()
{
    rtow-get
    rtow-cmake
    rtow-make
}



