Render PBO
===========

next
-----

#. is a near/far restricted raycast possible ? 

#. add switch target feature to daeviewgl : to enable easily doing raycasts wherever

#. avoid duplicated vertices/faces between chroma/cuda and opengl 

#. as threads get held back by the slowest within the block (or warp?) maybe better 
   to arrange blocks to cover rectangles of pixels rather than one pixel wide strips ?

   * slowest is the one that had to read the most nodes/triangles from memory ?



called twice 
-------------

Due to glut/glumpy window resize ?, the render gets called twice. 

For a successful small run::

    (chroma_env)delta:chroma_camera blyth$ python render_pbo.py 
    WARNING:env.geant4.geometry.collada.collada_to_chroma:setting parent_material to __dd__Materials__Vacuum0xaf1d298 as parent is None for node top.0 
    pixels (200, 200) launch LaunchSequence worksize 40000 max_blocks 128 threads_per_block 64 launches 5  
    [0] offset 0 grid (128, 1) block (64, 1, 1) 
    [1] offset 8192 grid (128, 1) block (64, 1, 1) 
    [2] offset 16384 grid (128, 1) block (64, 1, 1) 
    [3] offset 24576 grid (128, 1) block (64, 1, 1) 
    [4] offset 32768 grid (113, 1) block (64, 1, 1) 
    pixels (200, 200) launch LaunchSequence worksize 40000 max_blocks 128 threads_per_block 64 launches 5  
    [0] offset 0 grid (128, 1) block (64, 1, 1) 
    [1] offset 8192 grid (128, 1) block (64, 1, 1) 
    [2] offset 16384 grid (128, 1) block (64, 1, 1) 
    [3] offset 24576 grid (128, 1) block (64, 1, 1) 
    [4] offset 32768 grid (113, 1) block (64, 1, 1) 
    (chroma_env)delta:chroma_camera blyth$ 


glumpy frame size
--------------------

Keeping glumpy frame inset from figure at default of (0.9,0.9) leads to 
pixel offsets/complications.  When looking at timing pattern see half a block.
Avoid the issue and keep the frame at (1,1), so the pixels fill the figure window. 


daeviewgl integration
----------------------

Avoid wide view, and keep screen size small. As absolutely everything in the view gets visited it seems.::

    daeviewgl.py -t 8005 --with-chroma --cuda-profile --near 0.5 --size 640,480



CUDA Device Reset
-------------------

* http://stackoverflow.com/questions/7144195/cudadevicereset-for-multiple-gpus
* :google:`cudaDeviceReset driver API equivalent`

  * https://devtalk.nvidia.com/default/topic/686313/how-to-reset-cuda-error-in-driver-api/
  * need to destroy the context to recover, perhaps an atexit context.pop ?

  * http://lists.tiker.net/pipermail/pycuda/2011-June/003247.html

  * see `~/e/cuda` `cuda_info.sh` `cuda_info.py` for monitoring memory usage 



render_pbo kernel launch observations
---------------------------------------

#. changing alpha depth 3-10 seems not to have significant performance effect
#. cuda profile log gputimes in milliseconds and pycuda returned launch times in seconds agree quite well
#. pattern of launch times repeats, some pixels(directions through the geometry) are almost 1000x more expensive 
#. sometimes chroma has to split nodes, when that happens always get launch timeouts

#. such deaths related to (unsure of causality direction) persistent mis-behavior that can last for minutes

   * reducing activity on the machine clears it (closing tabs, windows)
   * maybe if anything goes wrong the GPU memory is not being properly cleaned up ? 



::

    2014-04-07 17:18:30,924 Optimization: Sufficient memory to move triangles onto GPU
    2014-04-07 17:18:30,936 Optimization: Sufficient memory to move vertices onto GPU
    2014-04-07 17:18:30,937 device usage:
    ----------
    nodes             2.8M  44.7M
    total                   44.7M
    ----------
    device total             2.1G
    device used              1.7G
    device free            408.3M

    2014-04-07 17:18:31,023 created PBORenderer 
    2014-04-07 17:18:31,024 scene init
    2014-04-07 17:18:31,037 scene draw
    2014-04-07 17:18:31,038 render Launch worksize (1024, 768) total 786432 max_blocks 1024 threads_per_block 64 launches 12 block (64, 1, 1)  
    2014-04-07 17:18:36,792 nprofile 12 nlaunch 12
    Launch worksize (1024, 768) total 786432 max_blocks 1024 threads_per_block 64 launches 12 block (64, 1, 1) 
    offset          0 count 65536 grid (1024, 1) block (64, 1, 1)  :         0.00681          6301.6        1540.587s 0.5 
    offset      65536 count 65536 grid (1024, 1) block (64, 1, 1)  :         0.01097         10930.8           5.675s 0.5 
    offset     131072 count 65536 grid (1024, 1) block (64, 1, 1)  :         0.03746         37415.4           8.493s 0.5 
    offset     196608 count 65536 grid (1024, 1) block (64, 1, 1)  :         0.36371        363682.8           8.011s 0.5 
    offset     262144 count 65536 grid (1024, 1) block (64, 1, 1)  :         0.59253        592511.9           8.837s 0.5 
    offset     327680 count 65536 grid (1024, 1) block (64, 1, 1)  :         0.79366        793652.1           9.258s 0.5 
    offset     393216 count 65536 grid (1024, 1) block (64, 1, 1)  :         0.97799        977982.1           5.761s 0.5 
    offset     458752 count 65536 grid (1024, 1) block (64, 1, 1)  :         1.10103       1101015.2           4.811s 0.5 
    offset     524288 count 65536 grid (1024, 1) block (64, 1, 1)  :         0.59855        598542.8           9.474s 0.5 
    offset     589824 count 65536 grid (1024, 1) block (64, 1, 1)  :         0.49433        494320.5          12.027s 0.5 
    offset     655360 count 65536 grid (1024, 1) block (64, 1, 1)  :         0.53871        538697.0           5.174s 0.5 
    offset     720896 count 65536 grid (1024, 1) block (64, 1, 1)  :         0.21149        211476.8           4.884s 0.5 
    2014-04-07 17:18:36,810 scene draw
    2014-04-07 17:18:36,810 render Launch worksize (1024, 768) total 786432 max_blocks 1024 threads_per_block 64 launches 12 block (64, 1, 1)  
    2014-04-07 17:18:42,116 nprofile 24 nlaunch 12
    Launch worksize (1024, 768) total 786432 max_blocks 1024 threads_per_block 64 launches 12 block (64, 1, 1) 
    offset          0 count 65536 grid (1024, 1) block (64, 1, 1)  :         0.00294          2747.1          11.575s 0.5 
    offset      65536 count 65536 grid (1024, 1) block (64, 1, 1)  :         0.00479          4778.2           4.144s 0.5 
    offset     131072 count 65536 grid (1024, 1) block (64, 1, 1)  :         0.01632         16301.1           8.594s 0.5 
    offset     196608 count 65536 grid (1024, 1) block (64, 1, 1)  :         0.19747        197449.9           8.445s 0.5 
    offset     262144 count 65536 grid (1024, 1) block (64, 1, 1)  :         0.45458        454571.2           5.197s 0.5 
    offset     327680 count 65536 grid (1024, 1) block (64, 1, 1)  :         0.74471        744698.5           3.977s 0.5 
    offset     393216 count 65536 grid (1024, 1) block (64, 1, 1)  :         0.96353        963522.1           7.007s 0.5 
    offset     458752 count 65536 grid (1024, 1) block (64, 1, 1)  :         1.08026       1080253.2           8.065s 0.5 
    offset     524288 count 65536 grid (1024, 1) block (64, 1, 1)  :         0.59160        591585.8           5.246s 0.5 
    offset     589824 count 65536 grid (1024, 1) block (64, 1, 1)  :         0.48775        487742.6           3.921s 0.5 
    offset     655360 count 65536 grid (1024, 1) block (64, 1, 1)  :         0.53199        531985.9           3.566s 0.5 
    offset     720896 count 65536 grid (1024, 1) block (64, 1, 1)  :         0.21126        211253.5           3.725s 0.5 
    (chroma_env)delta:render_pbo blyth$ 
    (chroma_env)delta:render_pbo blyth$ 



kernel pixel times figure
----------------------------

Change output pixels to reflect the cycles for each pixel::

    289     // PBO format BGRA as that is preferred by OpenGL
    290 
    291     /*
    292     pixels[idx] = blue ;
    293     pixels[idx+1] = green ;
    294     pixels[idx+2] = red ;
    295     pixels[idx+3] = a ;
    296     */
    297 
    298     int64_t cycles = clock64() - start ;
    299     unsigned int stime = (int) cycles >> 10 ;
    300 
    301    // if (threadIdx.x == 0 && blockIdx.x == 0) printf("cycles %d  stime %d \n", cycles, stime );
    302    
    303     pixels[idx] = stime ;
    304     pixels[idx+1] = stime ;
    305     pixels[idx+2] = stime ;
    306     pixels[idx+3] = stime ;


With image size of 1024,768 see that lines of pixels report the same time, across frame see 32 blocks
where all pixels have the same tone. 1024/32 = 32 (is that due to warp size 32 ?). 

Perhaps a figure of the maximum tricount along the line of pixels would match this.

    
   
tri count metric
------------------

PMT tri count hotspots very evident, despite the render only showing the plain outside of the radslabs.



::

   (chroma_env)delta:render_pbo blyth$ ./render_pbo.py --cuda-profile --alpha-depth 10 --kernel render_pbo --size 1024,768 --view B --kernel-flags 2,0 


::

    256     if( g_flags.x > 0){
    257 
    258         //int64_t metric = clock64() - start ;
    259         int metric = tri_count ;
    260         
    261         unsigned int shifted_metric = (int) metric >> g_flags.x ;
    262         
    263         pixels[idx]   = shifted_metric ;
    264         pixels[idx+1] = shifted_metric ;
    265         pixels[idx+2] = shifted_metric ;
    266         pixels[idx+3] = shifted_metric ;









profile shows first launch much more expensive
-------------------------------------------------

#. not seeing this anymore ?

::

    1314 method=[ memcpyHtoD ] gputime=[ 1.312 ] cputime=[ 2.064 ]
    1315 method=[ fill ] gputime=[ 12.544 ] cputime=[ 13.317 ] occupancy=[ 1.000 ]
    1316 method=[ memcpyHtoD ] gputime=[ 1.344 ] cputime=[ 6.096 ]
    1317 method=[ memcpyHtoD ] gputime=[ 1.184 ] cputime=[ 2.946 ]
    1318 method=[ memcpyHtoD ] gputime=[ 1.344 ] cputime=[ 2.574 ]
    1319 method=[ fill ] gputime=[ 12.608 ] cputime=[ 22.530 ] occupancy=[ 1.000 ]
    1320 method=[ render_pbo ] gputime=[ 4591701.500 ] cputime=[ 470.986 ] occupancy=[ 0.500 ]
    1321 method=[ render_pbo ] gputime=[ 155.456 ] cputime=[ 14.779 ] occupancy=[ 0.500 ]
    1322 method=[ render_pbo ] gputime=[ 155.232 ] cputime=[ 5.127 ] occupancy=[ 0.500 ]
    1323 method=[ render_pbo ] gputime=[ 156.288 ] cputime=[ 4.489 ] occupancy=[ 0.500 ]
    1324 method=[ render_pbo ] gputime=[ 139.744 ] cputime=[ 8.254 ] occupancy=[ 0.500 ]
    1325 method=[ fill ] gputime=[ 5.856 ] cputime=[ 25.364 ] occupancy=[ 1.000 ]
    1326 method=[ render_pbo ] gputime=[ 4221800.000 ] cputime=[ 6.858 ] occupancy=[ 0.500 ]
    1327 method=[ render_pbo ] gputime=[ 158.400 ] cputime=[ 14.441 ] occupancy=[ 0.500 ]
    1328 method=[ render_pbo ] gputime=[ 158.528 ] cputime=[ 4.998 ] occupancy=[ 0.500 ]
    1329 method=[ render_pbo ] gputime=[ 159.040 ] cputime=[ 5.541 ] occupancy=[ 0.500 ]
    1330 method=[ render_pbo ] gputime=[ 140.256 ] cputime=[ 8.515 ] occupancy=[ 0.500 ]





standard chroma cam
---------------------

::

    chroma-cam -F $DAE_NAME



