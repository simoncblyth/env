CUDA GDB
=========

Docs
----

* http://docs.nvidia.com/cuda/cuda-gdb/
* http://developer.download.nvidia.com/compute/cuda/2_1/cudagdb/CUDA_GDB_User_Manual.pdf

Extras
-------

* `info cuda threads`  long list of threads showing file and line numbers in the kernels

Debug Setup
--------------------------

Its more convenient to debug by

#. put rMBP into console mode, by logging in as username ">console"
#. ssh into rMPB from G4PB and run debug scripts like 3199.sh from there 
#. its real slow debugging, consider using smaller image sizes


Debug Example
---------------

::

    cd ~/e/chroma/chroma_camera
    ./3199.sh

    (chroma_env)delta:chroma_camera blyth$ cat 3199.sh
    #!/bin/bash -l
    chroma-
    cd $ENV_HOME/chroma/chroma_camera
    cuda-gdb --args python -m pycuda.debug simplecamera.py -s3199 -d3 -f10 --eye=0,1,0 --lookat=0,0,0 -G -o 3199_000.png



::


    [Launch of CUDA Kernel 0 (write_size<<<(1,1,1),(1,1,1)>>>) on Device 0]
    *** compiler output in /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/tmpsMRB8a
    [Launch of CUDA Kernel 1 (write_size<<<(1,1,1),(1,1,1)>>>) on Device 0]
    *** compiler output in /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/tmpPoL2DR
    [Launch of CUDA Kernel 2 (write_size<<<(1,1,1),(1,1,1)>>>) on Device 0]
    INFO:chroma:Optimization: Sufficient memory to move triangles onto GPU
    INFO:chroma:Optimization: Sufficient memory to move vertices onto GPU
    INFO:chroma:device usage:
    ----------
    nodes             2.8M  44.7M
    total                   44.7M
    ----------
    device total             2.1G
    device used            233.4M
    device free              1.9G

    *** compiler output in /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/tmpAmcMTC
    *** compiler output in /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/tmp6ewtAR
    *** compiler output in /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/tmpLWLa8A
    *** compiler output in /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/tmpjbg9nG
    [Launch of CUDA Kernel 3 (fill<<<(64,1,1),(128,1,1)>>>) on Device 0]
    [Launch of CUDA Kernel 4 (fill<<<(64,1,1),(128,1,1)>>>) on Device 0]
    [Launch of CUDA Kernel 5 (render<<<(4801,1,1),(64,1,1)>>>) on Device 0]
    ^Cwarning: (Internal error: pc 0x104199240 in read in psymtab, but not in symtab.)


    Program received signal SIGINT, Interrupt.
    [Switching focus to CUDA kernel 5, grid 6, block (48,0,0), thread (32,0,0), device 0, sm 1, warp 8, lane 0]
    render(int, float3 * @generic, float3 * @generic, Geometry * @generic, unsigned int, unsigned int * @generic, float * @generic, unsigned int * @generic, float4 * @generic)<<<(4801,1,1),(64,1,1)>>> (nthreads=307200, _origin=0x707680000, _direction=0x707a20000, 
        __val_paramg=0x700181800, alpha_depth=3, pixels=0x7090c0000, _dx=0x707dc0000, dxlen=0x708f80000, _color=0x708160000) at kernel.cu:144
    144         if (n < 1) {
    (cuda-gdb) info cuda threads
       BlockIdx ThreadIdx To BlockIdx ThreadIdx Count         Virtual PC           Filename  Line 
    Kernel 5
    *  (48,0,0)  (32,0,0)    (48,0,0)  (47,0,0)    16 0x0000000104199240          kernel.cu   144 
       (48,0,0)  (48,0,0)    (48,0,0)  (48,0,0)     1 0x0000000131e17980           linalg.h    39 
       (48,0,0)  (49,0,0)    (48,0,0)  (63,0,0)    15 0x0000000104199240          kernel.cu   144 
       (56,0,0)   (0,0,0)    (56,0,0)  (15,0,0)    16 0x0000000104199240          kernel.cu   144 
       (56,0,0)  (16,0,0)    (56,0,0)  (16,0,0)     1 0x00000001041868b8         geometry.h    37 
       (56,0,0)  (17,0,0)    (56,0,0)  (31,0,0)    15 0x0000000104199240          kernel.cu   144 
       (63,0,0)  (32,0,0)    (63,0,0)  (47,0,0)    16 0x0000000104199240          kernel.cu   144 
       (63,0,0)  (48,0,0)    (63,0,0)  (48,0,0)     1 0x0000000131e2a4d8           matrix.h   222 
       (63,0,0)  (49,0,0)    (63,0,0)  (63,0,0)    15 0x0000000104199240          kernel.cu   144 
       (71,0,0)   (0,0,0)    (71,0,0)  (15,0,0)    16 0x0000000104199240          kernel.cu   144 
       (71,0,0)  (16,0,0)    (71,0,0)  (16,0,0)     1 0x0000000131e45828   math_functions.h  8215 
       (71,0,0)  (17,0,0)    (71,0,0)  (31,0,0)    15 0x0000000104199240          kernel.cu   144 
       (78,0,0)  (32,0,0)    (78,0,0)  (47,0,0)    16 0x0000000104199240          kernel.cu   144 
       (78,0,0)  (48,0,0)    (78,0,0)  (48,0,0)     1 0x0000000104156310        intersect.h    56 
      ...
      (199,0,0)   (3,0,0)   (199,0,0)   (3,0,0)     1 0x0000000104155518        intersect.h    33 
      (199,0,0)   (4,0,0)   (199,0,0)   (5,0,0)     2 0x00000001041991c0          kernel.cu    90 
      (199,0,0)   (6,0,0)   (199,0,0)   (6,0,0)     1 0x0000000104155518        intersect.h    33 
      (199,0,0)   (7,0,0)   (199,0,0)   (8,0,0)     2 0x0000000104199230          kernel.cu    90 
      (199,0,0)   (9,0,0)   (199,0,0)   (9,0,0)     1 0x00000001041991c0          kernel.cu    90 
    ---Type <return> to continue, or q <return> to quit---q
    Quit
    (cuda-gdb) list
    139             } // loop over children, starting with first_child
    140
    141         } // while nodes on stack
    142         
    143
    144         if (n < 1) {
    145             pixels[id] = 0;
    146             return;
    147         }
    148
    (cuda-gdb) p n
    $1 = 3
    (cuda-gdb) p id
    $2 = 3104
    (cuda-gdb) 





info cuda threads
-------------------

From the manual::

    CUDA-GDB provides an additional command (info cuda threads) which displays 
    a summary of all CUDA threads that are currently resident on the GPU.  CUDA 
    threads are specified using the same syntax as described in Section 4.6 and are 
    summarized by grouping all contiguous threads that are stopped at the same 
    program location.  A sample display can be seen below: 
     
    <<<(0,0),(0,0,0)>>> ... <<<(0,0),(31,0,0)>>>  
    GPUBlackScholesCallPut () at blackscholes.cu:73 
    <<<(0,0),(32,0,0)>>> ... <<<(119,0),(0,0,0)>>> 
     GPUBlackScholesCallPut () at blackscholes.cu:72 
     
    The above example shows 32 threads (a warp) that have been advanced to line 73 of 
    blackscholes.cu, and the remainder of the resident threads stopped at line 72. 
    Since this summary only shows thread coordinates for the start and end range, it 
    may be unclear how many threads or blocks are actually within the displayed range.  
    This can be checked by printing the values of gridDim and/or blockDim. 
    CUDA-GDB also has the ability to display a full list of each individual thread that is 
    currently resident on the GPU by using the info cuda threads all command. 



kernel debug
-------------


::

    simon:cuda blyth$ grep STACK_SIZE *.*
    mesh.h:#define STACK_SIZE 1000
    mesh.h:    unsigned int child_ptr_stack[STACK_SIZE];
    mesh.h:    unsigned int nchild_ptr_stack[STACK_SIZE];
    mesh.h:     if (curr >= STACK_SIZE) {
    render.cu:    unsigned int child_ptr_stack[STACK_SIZE];
    render.cu:    unsigned int nchild_ptr_stack[STACK_SIZE];
    render.cu:          //if (curr >= STACK_SIZE) {


::

    (998,0,0)  (18,0,0)   (998,0,0)  (18,0,0)     1 0x0000000104199230          kernel.cu    90 
    (998,0,0)  (19,0,0)   (998,0,0)  (28,0,0)    10 0x0000000104151750        intersect.h    71 
    (998,0,0)  (29,0,0)   (998,0,0)  (29,0,0)     1 0x0000000104199230          kernel.cu    90 
    ---Type <return> to continue, or q <return> to quit---q
    Quit
    (cuda-gdb) list
    139             } // loop over children, starting with first_child
    140
    141         } // while nodes on stack
    142         
    143
    144         if (n < 1) {
    145             pixels[id] = 0;
    146             return;
    147         }
    148
    (cuda-gdb) p origin
    $4 = {x = -16566.293, y = -801040.938, z = -8842.5}
    (cuda-gdb) p direction
    $5 = {x = 0.740086973, y = -0.669951439, z = 0.0586207509}
    (cuda-gdb) p n
    $6 = 3
    (cuda-gdb) p distance
    $7 = 9562.18848
    (cuda-gdb) p STACK_SIZE
    No symbol "STACK_SIZE" in current context.
    (cuda-gdb) p child_ptr_stack
    $8 = {139, 1644, 509, 6978, 1898, 30875, 622018, 622033, 622063, 622078, 622178, 622193, 622208, 622343, 622493, 622523, 622538, 622553, 622568, 622583, 1692132, 1692147, 1692162, 1692177, 1692192, 1692207, 1692222, 1692514, 1692529, 1418963, 1419182, 1419197, 
      2109819, 2111777, 2111779, 2111794, 2111809, 2111824, 2095481, 1734707, 2653458, 2653473, 2653741, 2653756, 2653771, 2653996, 2656458, 2656473, 2656488, 2656713, 4281938739, 4281873459, 4281938995, 4281873203, 4281873458, 4282004788, 4282004788, 4281939251, 
      4281938996, 4282004532, 4281938996, 4282004532, 4281939253, 4281873202, 4281873202, 4281873203, 4281938995, 4281873458, 4281938740, 4281873202, 4281938739, 4282004532, 4281938995, 4282004533, 4282004787, 4282004788, 4282004788, 4282004531, 4281938995, 
      4281873458, 4281873459, 4281939251, 4281938996, 4282004532, 4281938996, 4281938739, 4281873460, 4281939253, 4282004532, 4282004789, 4282070325, 4282004789, 4282005044, 4282004788, 4282004787, 4281873203, 4281938739, 4281938995, 4281938995, 4281938995, 
      4282004532, 4281873458, 4281938739, 4282070325, 4282004788, 4282004532, 4281939251, 4282004788, 4281938996, 4282070324, 4282004788, 4281939251, 4281938996, 4281939251, 4281939252, 4282004532, 4281939253, 4281938996, 4281873460, 4282005046, 4282070325, 
      4282004789, 4282070325, 4282004789, 4282005044, 4282070581, 4282070580, 4282004532, 4281938995, 4282004532, 4281939251, 4282004532, 4281938996, 4281938995, 4281938995, 4282004788, 4282070324, 4282070325, 4282070581, 4282070580, 4282070325, 4282004788, 
      4282070325, 4281938996, 4282004532, 4281939253, 4282004788, 4282004789, 4282005044, 4281938996, 4281939251, 4282070580, 4282005045, 4282070837, 4282070582, 4282136118, 4282070582, 4282070325, 4282005046, 4281873204, 4281938995, 4281807668, 4281938994, 
      4281872947, 4281938739, 4281873203, 4281938739, 4282070068, 4281938996, 4282004532, 4281938740, 4282004787, 4281938997, 4282070324, 4281938997, 4281938997, 4282004787, 4281938740, 4281938994, 4281873458, 4281873204, 4281939251, 4281938996, 4282004789, 
      4282005044, 4281939253, 4281939251, 4282004787, 4282004789, 4282070580, 4282004790, 4281938739, 4281873203, 4282004532, 4281938739, 4282004787, 4281873461, 4281938995, 4281873204, 4282004790...}
    (cuda-gdb) p nchild_ptr_stack
    $9 = {2, 2, 7, 2, 6, 3, 15, 15, 15, 15, 15, 15, 15, 8, 15 <repeats 17 times>, 2, 15, 2, 15 <repeats 16 times>, 4282267703, 4282333495, 4282333241, 4282333752, 4282399033, 4282070581, 4282070581, 4282136118, 4282070581, 4282136118, 4282070581, 4282136118, 
      4282070580, 4282202166, 4282267447, 4282136374, 4282201911, 4282136374, 4282201910, 4282201911, 4282202167, 4282070582, 4282070837, 4282136374, 4282136374, 4282136375, 4282136374, 4282070582, 4282136118, 4282267703, 4282202168, 4282201911, 4282136375, 
      4282202166, 4282202167, 4282202423, 4282202167, 4282070581, 4282136118, 4282136373, 4282201911, 4282136374, 4282136374, 4282070581, 4282136117, 4282201911, 4282136374, 4282267703, 4282202167, 4282267704, 4282202166, 4282201911, 4282136374, 4282136118, 
      4282070839, 4282201911, 4282136375, 4282136630, 4282136375, 4282136373, 4282136374, 4282202167, 4282202166, 4282267960, 4282267959, 4282202168, 4282267704, 4282202168, 4282267703, 4282070837, 4282136118, 4282136374, 4282201911, 4282136374, 4282201910, 
      4282136117, 4282136374, 4282267703, 4282267703, 4282202167, 4282202167, 4282267703, 4282202166, 4282267704, 4282202167, 4282201910, 4282136375, 4282201911, 4282136631, 4282202166, 4282202167, 4282136630, 4282136374, 4282267960, 4282267959, 4282202168, 
      4282202423, 4282202168, 4282267704, 4282267961, 4282333496, 4282201910, 4282136374, 4282202167, 4282201910, 4282201911, 4282202166, 4282201911, 4282136373, 4282202167, 4282267704, 4282267959, 4282333240, 4282267960, 4282333496, 4282202167, 4282267703, 
      4282136375, 4282136630, 4282202167, 4282202423, 4282202168, 4282267703, 4282136375, 4282201911, 4282267704, 4282202424, 4282333496, 4282267961, 4282268216, 4282267961, 4282267959, 4282267960, 4282136118, 4282201910, 4282070325, 4282136118, 4282070583, 
      4282136373, 4282070583, 4282201910, 4282267703, 4282202168, 4282267702, 4282136376, 4282201911, 4282136375, 4282267703, 4282201911, 4282136630, 4282136375, 4282136373, 4282070838, 4282136375, 4282136373, 4282201912, 4282202166, 4282267959, 4282267705, 
      4282267958, 4282202168, 4282202168, 4282202166...}
    (cuda-gdb) 



    (cuda-gdb) p sg
    $11 = {vertices = 0x706600000, triangles = 0x704980000, material_codes = 0x700240000, colors = 0x700bc0000, primary_nodes = 0x701ec0000, extra_nodes = 0x202b00000, materials = 0x70015b000, surfaces = 0x700181600, world_origin = {x = -2400000, y = -2400000, 
        z = -2400000}, world_scale = 73.2444229, nprimary_nodes = 2794974}
    (cuda-gdb) p g
    $12 = <value optimized out>
    (cuda-gdb) p id
    $13 = 56864
    (cuda-gdb) p root
    $14 = {lower = {x = -2400000, y = -2400000, z = -2400000}, upper = {x = 2400073.5, y = 2400073.5, z = 2400073.5}, child = 1, nchild = 2}
    (cuda-gdb) p neg_origin_inv_dir
    $15 = {x = 22384.252, y = -1195670.12, z = 150842.484}
    (cuda-gdb) p inv_dir
    $16 = {x = 1.35119259, y = -1.4926455, z = 17.0588055}
    (cuda-gdb) p count
    $17 = <value optimized out>
    (cuda-gdb) p tri_count
    $18 = <value optimized out>
    (cuda-gdb) p alpha_depth
    $19 = 3
    (cuda-gdb) p _dx
    $20 = (@generic float * @parameter) 0x707dc0000
    (cuda-gdb) p dx
    $21 = <value optimized out>
    (cuda-gdb) p _color
    $22 = (@generic float4 * @parameter) 0x708160000
    (cuda-gdb) p color_a
    $23 = (@generic float4 * @register) 0x7083fa600




::

    kernel.cu    90 
       (998,0,0)   (9,0,0)   (998,0,0)  (13,0,0)     5 0x0000000104151750        intersect.h    71 
       (998,0,0)  (14,0,0)   (998,0,0)  (14,0,0)     1 0x0000000104199230          kernel.cu    90 
       (998,0,0)  (15,0,0)   (998,0,0)  (15,0,0)     1 0x0000000104151750        intersect.h    71 
       (998,0,0)  (16,0,0)   (998,0,0)  (17,0,0)     2 0x0000000104199240          kernel.cu   144 
       (998,0,0)  (18,0,0)   (998,0,0)  (18,0,0)     1 0x0000000104199230          kernel.cu    90 
       (998,0,0)  (19,0,0)   (998,0,0)  (28,0,0)    10 0x0000000104151750        intersect.h    71 
       (998,0,0)  (29,0,0)   (998,0,0)  (29,0,0)     1 0x0000000104199230          kernel.cu    90 
    ---Type <return> to continue, or q <return> to quit--- q
    Quit
    (cuda-gdb) info cuda state
    Unrecognized option: 'state'.
    (cuda-gdb) bt
    #0  render(int, float3 * @generic, float3 * @generic, Geometry * @generic, unsigned int, unsigned int * @generic, float * @generic, unsigned int * @generic, float4 * @generic)<<<(4801,1,1),(64,1,1)>>> (nthreads=307200, _origin=0x707680000, _direction=0x707a20000, 
        __val_paramg=0x700181800, alpha_depth=3, pixels=0x7090c0000, _dx=0x707dc0000, dxlen=0x708f80000, _color=0x708160000) at kernel.cu:144
    (cuda-gdb) list
    149         dxlen[id] = n;
    150
    151         float scale = 1.0f;
    152         float fr = 0.0f;
    153         float fg = 0.0f;
    154         float fb = 0.0f;
    155         for (int i=0; i < n; i++) {
    156             float alpha = color_a[i].w;
    157
    158             fr += scale*color_a[i].x*alpha;
    (cuda-gdb) c
    Continuing.
    ^C
    Program received signal SIGINT, Interrupt.
    [Switching focus to CUDA kernel 5, grid 6, block (1623,0,0), thread (32,0,0), device 0, sm 1, warp 4, lane 0]
    render(int, float3 * @generic, float3 * @generic, Geometry * @generic, unsigned int, unsigned int * @generic, float * @generic, unsigned int * @generic, float4 * @generic)<<<(4801,1,1),(64,1,1)>>> (nthreads=307200, _origin=0x707680000, _direction=0x707a20000, 
        __val_paramg=0x700181800, alpha_depth=3, pixels=0x7090c0000, _dx=0x707dc0000, dxlen=0x708f80000, _color=0x708160000) at kernel.cu:144
    144         if (n < 1) {
    (cuda-gdb) bt
    #0  render(int, float3 * @generic, float3 * @generic, Geometry * @generic, unsigned int, unsigned int * @generic, float * @generic, unsigned int * @generic, float4 * @generic)<<<(4801,1,1),(64,1,1)>>> (nthreads=307200, _origin=0x707680000, _direction=0x707a20000, 
        __val_paramg=0x700181800, alpha_depth=3, pixels=0x7090c0000, _dx=0x707dc0000, dxlen=0x708f80000, _color=0x708160000) at kernel.cu:144
    (cuda-gdb) p id
    $26 = 103904
    (cuda-gdb) thread
    Focus not set on any host thread.
    (cuda-gdb) print blockIdx
    $27 = {x = 1623, y = 0, z = 0}
    (cuda-gdb) print threadIdx
    $28 = {x = 32, y = 0, z = 0}
    (cuda-gdb) print blockDim
    $29 = {x = 64, y = 1, z = 1}
    (cuda-gdb) print gridDim
    $30 = {x = 4801, y = 1, z = 1}
    (cuda-gdb) p nthreads
    $31 = 307200
    (cuda-gdb) thread <<<0>>>
    A syntax error in expression, near `<<<0>>>'.
    (cuda-gdb) c
    Continuing.
    ^C[New Thread 0x297b of process 6669]
    warning: (Internal error: pc 0x10412b390 in read in psymtab, but not in symtab.)


    Program received signal SIGINT, Interrupt.
    [Switching focus to CUDA kernel 5, grid 6, block (2400,0,0), thread (0,0,0), device 0, sm 0, warp 12, lane 0]
    0x000000010412b390 in intersect_node(Geometry * @generic, const float3 * @generic, const float3 * @generic, const Node * @generic, const float) (g=0x1000000, neg_origin_inv_dir=<value optimized out>, inv_dir=<value optimized out>, node=<value optimized out>, 
        min_distance=<value optimized out>) at mesh.h:32
    32              return false;
    (cuda-gdb) list
    27                  return false;
    28
    29              return true;
    30          }
    31          else {
    32              return false;
    33          }
    34      }
    35
    36      /* Finds the intersection between a ray and `geometry`. If the ray does
    (cuda-gdb) p id
    No symbol "id" in current context.
    (cuda-gdb) bt
    #0  0x000000010412b390 in intersect_node(Geometry * @generic, const float3 * @generic, const float3 * @generic, const Node * @generic, const float) (g=0x1000000, neg_origin_inv_dir=<value optimized out>, inv_dir=<value optimized out>, node=<value optimized out>, 
        min_distance=<value optimized out>) at mesh.h:32
    #1  0x0000000104198158 in render(int, float3 * @generic, float3 * @generic, Geometry * @generic, unsigned int, unsigned int * @generic, float * @generic, unsigned int * @generic, float4 * @generic)<<<(4801,1,1),(64,1,1)>>> (nthreads=307200, _origin=0x707680000, 
        _direction=0x707a20000, __val_paramg=0x700181800, alpha_depth=3, pixels=0x7090c0000, _dx=0x707dc0000, dxlen=0x708f80000, _color=0x708160000) at kernel.cu:94
    (cuda-gdb) u
    warning: (Internal error: pc 0x104198158 in read in psymtab, but not in symtab.)

    render(int, float3 * @generic, float3 * @generic, Geometry * @generic, unsigned int, unsigned int * @generic, float * @generic, unsigned int * @generic, float4 * @generic)<<<(4801,1,1),(64,1,1)>>> (nthreads=307200, _origin=0x707680000, _direction=0x707a20000, 
        __val_paramg=0x700181800, alpha_depth=3, pixels=0x7090c0000, _dx=0x707dc0000, dxlen=0x708f80000, _color=0x708160000) at kernel.cu:90
    90              for (unsigned int i=first_child; i < first_child + nchild; i++) {
    (cuda-gdb) p id
    $32 = 153600
    (cuda-gdb) p first_child
    $33 = 1671291
    (cuda-gdb) p nchild
    $34 = 15
    (cuda-gdb) p curr
    $35 = 19
    (cuda-gdb) p g
    $36 = (Geometry * @generic) 0x1000000
    (cuda-gdb) p node
    $37 = {lower = {x = -17725.25, y = -802099.625, z = -7910.5}, upper = {x = -17578.75, y = -801953.125, z = -7690.75}, child = 268267, nchild = 0}
    (cuda-gdb) 




::

    (cuda-gdb) info threads
      7 Thread 0x1553 of process 6669  0x00007fff8a183a1a in mach_msg_trap () from /usr/lib/system/libsystem_kernel.dylib
      6 Thread 0x2703 of process 6669  0x00007fff8a183a1a in mach_msg_trap () from /usr/lib/system/libsystem_kernel.dylib
      3 Thread 0x1623 of process 6669  0x00007fff8a188662 in kevent64 () from /usr/lib/system/libsystem_kernel.dylib
      2 Thread 0x1807 of process 6669  0x00007fff8a187a3a in __semwait_signal () from /usr/lib/system/libsystem_kernel.dylib
    * 1 Thread 0x2303 of process 6669  0x0000000103bce666 in cudbgMain () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    (cuda-gdb) bt
    #0  render(int, float3 * @generic, float3 * @generic, Geometry * @generic, unsigned int, unsigned int * @generic, float * @generic, unsigned int * @generic, float4 * @generic)<<<(4801,1,1),(64,1,1)>>> (nthreads=307200, _origin=0x707680000, _direction=0x707a20000, 
        __val_paramg=0x700181800, alpha_depth=3, pixels=0x7090c0000, _dx=0x707dc0000, dxlen=0x708f80000, _color=0x708160000) at kernel.cu:144
    (cuda-gdb) thread 1
    [Switching to thread 1 (Thread 0x2303 of process 6669)]#0  0x0000000103bce666 in cudbgMain () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    (cuda-gdb) bt
    #0  0x0000000103bce666 in cudbgMain () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    #1  0x0000000103b730c9 in cuGraphicsGLRegisterImage () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    #2  0x0000000103a8a1f3 in cuGraphicsGLRegisterImage () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    #3  0x0000000103b75e66 in cuGraphicsGLRegisterImage () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    #4  0x0000000103b75fd1 in cuGraphicsGLRegisterImage () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    #5  0x0000000103b61c1e in cuGraphicsGLRegisterImage () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    #6  0x0000000103b61f0d in cuGraphicsGLRegisterImage () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    #7  0x0000000103b571e5 in cuGraphicsGLRegisterImage () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    #8  0x0000000103a7eb51 in cuGraphicsGLRegisterImage () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    #9  0x0000000103a8224f in cuGraphicsGLRegisterImage () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    #10 0x0000000103a7105d in cuMemcpyDtoH_v2 () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    #11 0x0000000101816ba4 in (anonymous namespace)::py_memcpy_dtoh(pycudaboost::python::api::object, unsigned long long) () from /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/_driver.so
    #12 0x0000000101839e1d in pycudaboost::python::detail::caller_arity<2u>::impl<void (*)(pycudaboost::python::api::object, unsigned long long), pycudaboost::python::default_call_policies, pycudaboost::mpl::vector3<void, pycudaboost::python::api::object, unsigned long long> >::operator()(_object*, _object*) () from /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/_driver.so
    #13 0x0000000101869d4e in pycudaboost::python::objects::function::call(_object*, _object*) const () from /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/_driver.so
    #14 0x000000010186bf7a in pycudaboost::detail::function::void_function_ref_invoker0<pycudaboost::python::objects::(anonymous namespace)::bind_return, void>::invoke(pycudaboost::detail::function::function_buffer&) ()
       from /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/_driver.so
    #15 0x00000001018799f3 in pycudaboost::python::detail::exception_handler::operator()(pycudaboost::function0<void> const&) const () from /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/_driver.so
    #16 0x0000000101851f76 in pycudaboost::detail::function::function_obj_invoker2<pycudaboost::_bi::bind_t<bool, pycudaboost::python::detail::translate_exception<pycuda::error, void (*)(pycuda::error const&)>, pycudaboost::_bi::list3<pycudaboost::arg<1>, pycudaboost::arg<2>, pycudaboost::_bi::value<void (*)(pycuda::error const&)> > >, bool, pycudaboost::python::detail::exception_handler const&, pycudaboost::function0<void> const&>::invoke(pycudaboost::detail::function::function_buffer&, pycudaboost::python::detail::exception_handler const&, pycudaboost::function0<void> const&) () from /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/_driver.so
    #17 0x0000000101879783 in pycudaboost::python::handle_exception_impl(pycudaboost::function0<void>) () from /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/_driver.so
    #18 0x000000010186b963 in function_call () from /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/_driver.so
    #19 0x0000000100011665 in PyObject_Call () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #20 0x00000001000a60b4 in PyEval_EvalFrameEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #21 0x00000001000a2076 in PyEval_EvalCodeEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #22 0x00000001000a8f36 in fast_function () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #23 0x00000001000a528b in PyEval_EvalFrameEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #24 0x00000001000a8ed2 in fast_function () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #25 0x00000001000a528b in PyEval_EvalFrameEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #26 0x00000001000a2076 in PyEval_EvalCodeEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #27 0x00000001000a8f36 in fast_function () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #28 0x00000001000a528b in PyEval_EvalFrameEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #29 0x00000001000a8ed2 in fast_function () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #30 0x00000001000a528b in PyEval_EvalFrameEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #31 0x00000001000a2076 in PyEval_EvalCodeEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #32 0x00000001000a19a6 in PyEval_EvalCode () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #33 0x00000001000c9611 in PyRun_FileExFlags () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #34 0x000000010009dfe6 in builtin_execfile () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #35 0x00000001000a4010 in PyEval_EvalFrameEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #36 0x00000001000a2076 in PyEval_EvalCodeEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #37 0x00000001000a6752 in PyEval_EvalFrameEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #38 0x00000001000a2076 in PyEval_EvalCodeEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #39 0x00000001000a8f36 in fast_function () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #40 0x00000001000a528b in PyEval_EvalFrameEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #41 0x00000001000a2076 in PyEval_EvalCodeEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #42 0x00000001000350c6 in function_call () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #43 0x0000000100011665 in PyObject_Call () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #44 0x00000001000dd131 in RunModule () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #45 0x00000001000dcc12 in Py_Main () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #46 0x00007fff904935fd in start () from /usr/lib/system/libdyld.dylib
    #47 0x00007fff904935fd in start () from /usr/lib/system/libdyld.dylib
    #48 0x0000000000000000 in ?? ()
    (cuda-gdb) 


::

    (cuda-gdb) c
    Continuing.
    ^Cwarning: (Internal error: pc 0x104199240 in read in psymtab, but not in symtab.)


    Program received signal SIGINT, Interrupt.
    [Switching focus to CUDA kernel 5, grid 6, block (2403,0,0), thread (32,0,0), device 0, sm 1, warp 2, lane 0]
    render(int, float3 * @generic, float3 * @generic, Geometry * @generic, unsigned int, unsigned int * @generic, float * @generic, unsigned int * @generic, float4 * @generic)<<<(4801,1,1),(64,1,1)>>> (nthreads=307200, _origin=0x707680000, _direction=0x707a20000, 
        __val_paramg=0x700181800, alpha_depth=3, pixels=0x7090c0000, _dx=0x707dc0000, dxlen=0x708f80000, _color=0x708160000) at kernel.cu:144
    144         if (n < 1) {
    (cuda-gdb) cuda device sm warp lane block thread
    block (2403,0,0), thread (32,0,0), device 0, sm 1, warp 2, lane 0
    (cuda-gdb) cuda kernel block thread
    kernel 5, block (2403,0,0), thread (32,0,0)
    (cuda-gdb) cuda kernel
    kernel 5
    (cuda-gdb) cuda device 0 sm 1 warp 2 lane 3
    [Switching focus to CUDA kernel 5, grid 6, block (2403,0,0), thread (35,0,0), device 0, sm 1, warp 2, lane 3]
    144         if (n < 1) {
    (cuda-gdb) list
    139             } // loop over children, starting with first_child
    140
    141         } // while nodes on stack
    142         
    143
    144         if (n < 1) {
    145             pixels[id] = 0;
    146             return;
    147         }
    148
    (cuda-gdb) p id
    $40 = 153827
    (cuda-gdb) p n
    $41 = 3
    (cuda-gdb) c
    Continuing.



Stopping when the fans spin up, is pointing at device to host memcopy::

    (cuda-gdb) info threads
      7 Thread 0x1553 of process 6669  0x00007fff8a183a1a in mach_msg_trap () from /usr/lib/system/libsystem_kernel.dylib
      6 Thread 0x2703 of process 6669  0x00007fff8a183a1a in mach_msg_trap () from /usr/lib/system/libsystem_kernel.dylib
      3 Thread 0x1623 of process 6669  0x00007fff8a188662 in kevent64 () from /usr/lib/system/libsystem_kernel.dylib
      2 Thread 0x1807 of process 6669  0x00007fff8a187a3a in __semwait_signal () from /usr/lib/system/libsystem_kernel.dylib
    * 1 Thread 0x2303 of process 6669  0x0000000103b757a4 in cuGraphicsGLRegisterImage () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    (cuda-gdb) thread 1
    [Switching to thread 1 (Thread 0x2303 of process 6669)]#0  0x0000000103b757a4 in cuGraphicsGLRegisterImage () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    (cuda-gdb) bt
    #0  0x0000000103b757a4 in cuGraphicsGLRegisterImage () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    #1  0x0000000103b75ce0 in cuGraphicsGLRegisterImage () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    #2  0x0000000103b75fd1 in cuGraphicsGLRegisterImage () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    #3  0x0000000103b61c1e in cuGraphicsGLRegisterImage () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    #4  0x0000000103b61f0d in cuGraphicsGLRegisterImage () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    #5  0x0000000103b571e5 in cuGraphicsGLRegisterImage () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    #6  0x0000000103a7eb51 in cuGraphicsGLRegisterImage () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    #7  0x0000000103a8224f in cuGraphicsGLRegisterImage () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    #8  0x0000000103a7105d in cuMemcpyDtoH_v2 () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    #9  0x0000000101816ba4 in (anonymous namespace)::py_memcpy_dtoh(pycudaboost::python::api::object, unsigned long long) () from /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/_driver.so
    #10 0x0000000101839e1d in pycudaboost::python::detail::caller_arity<2u>::impl<void (*)(pycudaboost::python::api::object, unsigned long long), pycudaboost::python::default_call_policies, pycudaboost::mpl::vector3<void, pycudaboost::python::api::object, unsigned long long> >::operator()(_object*, _object*) () from /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/_driver.so
    #11 0x0000000101869d4e in pycudaboost::python::objects::function::call(_object*, _object*) const () from /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/_driver.so
    #12 0x000000010186bf7a in pycudaboost::detail::function::void_function_ref_invoker0<pycudaboost::python::objects::(anonymous namespace)::bind_return, void>::invoke(pycudaboost::detail::function::function_buffer&) ()
       from /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/_driver.so
    #13 0x00000001018799f3 in pycudaboost::python::detail::exception_handler::operator()(pycudaboost::function0<void> const&) const () from /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/_driver.so
    #14 0x0000000101851f76 in pycudaboost::detail::function::function_obj_invoker2<pycudaboost::_bi::bind_t<bool, pycudaboost::python::detail::translate_exception<pycuda::error, void (*)(pycuda::error const&)>, pycudaboost::_bi::list3<pycudaboost::arg<1>, pycudaboost::arg<2>, pycudaboost::_bi::value<void (*)(pycuda::error const&)> > >, bool, pycudaboost::python::detail::exception_handler const&, pycudaboost::function0<void> const&>::invoke(pycudaboost::detail::function::function_buffer&, pycudaboost::python::detail::exception_handler const&, pycudaboost::function0<void> const&) () from /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/_driver.so
    #15 0x0000000101879783 in pycudaboost::python::handle_exception_impl(pycudaboost::function0<void>) () from /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/_driver.so
    #16 0x000000010186b963 in function_call () from /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/_driver.so
    #17 0x0000000100011665 in PyObject_Call () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #18 0x00000001000a60b4 in PyEval_EvalFrameEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #19 0x00000001000a2076 in PyEval_EvalCodeEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #20 0x00000001000a8f36 in fast_function () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #21 0x00000001000a528b in PyEval_EvalFrameEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #22 0x00000001000a8ed2 in fast_function () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #23 0x00000001000a528b in PyEval_EvalFrameEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #24 0x00000001000a2076 in PyEval_EvalCodeEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #25 0x00000001000a8f36 in fast_function () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #26 0x00000001000a528b in PyEval_EvalFrameEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #27 0x00000001000a8ed2 in fast_function () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #28 0x00000001000a528b in PyEval_EvalFrameEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #29 0x00000001000a2076 in PyEval_EvalCodeEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #30 0x00000001000a19a6 in PyEval_EvalCode () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #31 0x00000001000c9611 in PyRun_FileExFlags () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #32 0x000000010009dfe6 in builtin_execfile () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #33 0x00000001000a4010 in PyEval_EvalFrameEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #34 0x00000001000a2076 in PyEval_EvalCodeEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #35 0x00000001000a6752 in PyEval_EvalFrameEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #36 0x00000001000a2076 in PyEval_EvalCodeEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #37 0x00000001000a8f36 in fast_function () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #38 0x00000001000a528b in PyEval_EvalFrameEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #39 0x00000001000a2076 in PyEval_EvalCodeEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #40 0x00000001000350c6 in function_call () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #41 0x0000000100011665 in PyObject_Call () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #42 0x00000001000dd131 in RunModule () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #43 0x00000001000dcc12 in Py_Main () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #44 0x00007fff904935fd in start () from /usr/lib/system/libdyld.dylib
    #45 0x00007fff904935fd in start () from /usr/lib/system/libdyld.dylib
    #46 0x0000000000000000 in ?? ()
    (cuda-gdb) 


::

    (cuda-gdb) c
    Continuing.
    ^Cwarning: (Internal error: pc 0x104199240 in read in psymtab, but not in symtab.)


    Program received signal SIGINT, Interrupt.
    [Switching focus to CUDA kernel 5, grid 6, block (2403,0,0), thread (35,0,0), device 0, sm 1, warp 2, lane 3]
    render(int, float3 * @generic, float3 * @generic, Geometry * @generic, unsigned int, unsigned int * @generic, float * @generic, unsigned int * @generic, float4 * @generic)<<<(4801,1,1),(64,1,1)>>> (nthreads=307200, _origin=0x707680000, _direction=0x707a20000, 
        __val_paramg=0x700181800, alpha_depth=3, pixels=0x7090c0000, _dx=0x707dc0000, dxlen=0x708f80000, _color=0x708160000) at kernel.cu:144
    144         if (n < 1) {
    (cuda-gdb) info contexts
    Undefined info command: "contexts".  Try "help info".
    (cuda-gdb) info cuda contexts
         Context Dev    State 
    * 0x10097d200   0   active 
    (cuda-gdb) info cuda blocks
        BlockIdx To BlockIdx Count   State 
    Kernel 5
    * (2403,0,0)  (2403,0,0)     1 running 
    (cuda-gdb) info cuda threads
        BlockIdx ThreadIdx To BlockIdx ThreadIdx Count         Virtual PC   Filename  Line 
    Kernel 5
    * (2403,0,0)  (32,0,0)  (2403,0,0)  (47,0,0)    16 0x0000000104199240  kernel.cu   144 
      (2403,0,0)  (48,0,0)  (2403,0,0)  (48,0,0)     1 0x0000000131e41a30 geometry.h    10 
      (2403,0,0)  (49,0,0)  (2403,0,0)  (63,0,0)    15 0x0000000104199240  kernel.cu   144 
    (cuda-gdb) info cuda kernels
      Kernel Parent Dev Grid Status   SMs Mask    GridDim BlockDim Invocation 
    *      5      -   0    6 Active 0x00000002 (4801,1,1) (64,1,1) render(nthreads=307200, _origin=0x707680000, _direction=0x707a20000, __val_paramg=0x700181800, alpha_depth=3, pixels=0x7090c0000, _dx=0x707dc0000, dxlen=0x708f80000, _color=0x708160000) 
    (cuda-gdb) c
    Continuing.
    INFO:chroma:saving screen to 3199_000.png 
    [New Thread 0x391b of process 6669]
    [Context Pop of context 0x10097d200 on Device 0]
    [Termination of CUDA Kernel 5 (render<<<(4801,1,1),(64,1,1)>>>) on Device 0]
    [Context Push of context 0x10097d200 on Device 0]
    [Context Pop of context 0x10097d200 on Device 0]
    [Context Push of context 0x10097d200 on Device 0]
    [Context Pop of context 0x10097d200 on Device 0]
    [Context Push of context 0x10097d200 on Device 0]
    [Context Pop of context 0x10097d200 on Device 0]

    Before eliding...

        simon:cuda blyth$ grep Context debug.rst | wc -l   
            1078

    [Context Push of context 0x10097d200 on Device 0]
    [Context Pop of context 0x10097d200 on Device 0]
    [Context Push of context 0x10097d200 on Device 0]
    [Context Pop of context 0x10097d200 on Device 0]
    [Context Push of context 0x10097d200 on Device 0]
    [Context Pop of context 0x10097d200 on Device 0]
    [Context Push of context 0x10097d200 on Device 0]
    [Context Pop of context 0x10097d200 on Device 0]
    [Context Push of context 0x10097d200 on Device 0]
    [Context Pop of context 0x10097d200 on Device 0]

    Program exited normally.
    [Termination of CUDA Kernel 4 (fill<<<(64,1,1),(128,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 3 (fill<<<(64,1,1),(128,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 2 (write_size<<<(1,1,1),(1,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 1 (write_size<<<(1,1,1),(1,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 0 (write_size<<<(1,1,1),(1,1,1)>>>) on Device 0]
    (cuda-gdb) 




PyCUDA Version
----------------

::

    (chroma_env)delta:chroma_camera blyth$ python -c "import pycuda ; print pycuda.VERSION "
    (2013, 1, 1)
    (chroma_env)delta:chroma_camera blyth$ python -c "import pycuda ; print pycuda.VERSION_STATUS "

    (chroma_env)delta:chroma_camera blyth$ python -c "import pycuda ; print pycuda.VERSION_TEXT "
    2013.1.1


pudb : Console based python debugger
-------------------------------------

Referenced from PyCUDA FAQ

* https://pypi.python.org/pypi/pudb


PyCUDA FAQ Extracts
----------------------


* http://wiki.tiker.net/PyCuda/FrequentlyAskedQuestions

Below Section of FAQ may be related to issue
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Downgrading of former cleanup failures to warnings in 0.93 smells like
it could be contributing to the GPU panics I see on OSX, which are 
always preceeded by screen fulls of::


My program terminates after a launch failure. Why?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* This should not be an issue any more with 0.93 and later, where cleanup failures have been downgraded to warnings.

You're probably seeing something like this::

    Traceback (most recent call last):
      File "fail.py", line 32, in <module>
        cuda.memcpy_dtoh(a_doubled, a_gpu)
    RuntimeError: cuMemcpyDtoH failed: launch failed
    terminate called after throwing an instance of 'std::runtime_error'
      what():  cuMemFree failed: launch failed
    zsh: abort      python fail.py

What's going on here? First of all, recall that launch failures in CUDA are
asynchronous. So the actual traceback does not point to the failed kernel
launch, it points to the next CUDA request after the failed kernel.

Next, as far as I can tell, a CUDA context becomes invalid after a launch
failure, and all following CUDA calls in that context fail. Now, that includes
cleanup (see the cuMemFree in the traceback?) that PyCUDA tries to perform
automatically. Here, a bit of PyCUDA's C++ heritage shows through. While
performing cleanup, we are processing an exception (the launch failure reported
by cuMemcpyDtoH). If another exception occurs during exception processing, C++
gives up and aborts the program with a message.

In principle, this could be handled better. If you're willing to dedicate time
to this, I'll likely take your patch.




