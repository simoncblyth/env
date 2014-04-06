Render PBO
===========



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



profile shows first launch much more expensive
---------------------------------------------

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




array size bug
----------------

::

    chroma-cam -F $DAE_NAME


::

    npixels 589824 width 1024 height  576     1024*576 = 589824

    pos 1769472    # size of np array is element count     589824 * 3 = 1769472
    [[       0. -8313844.        0.]
     [       0. -8313844.        0.]
     [       0. -8313844.        0.]
     ..., 
     [       0. -8313844.        0.]
     [       0. -8313844.        0.]
     [       0. -8313844.        0.]]

    dir 1769472 
    [[-0.64897239  0.66751446  0.36504697]
     [-0.64927236  0.667823    0.36394759]
     [-0.6495717   0.66813089  0.36284669]
     ..., 
     [ 0.64913628  0.66898965 -0.36204274]
     [ 0.64883741  0.66868165 -0.36314579]
     [ 0.64853792  0.66837299 -0.36424732]]


    max_alpha_depth 10 
    pos.size   589824 
    dx.size    5898240 
    dxlen.size 589824 
    color.size 5898240 

::

     23 
     24         self.dx = ga.empty(max_alpha_depth*self.pos.size, dtype=np.float32)
     25         self.color = ga.empty(self.dx.size, dtype=ga.vec.float4)
     26         self.dxlen = ga.zeros(self.pos.size, dtype=np.uint32)
     27 
     28         print "max_alpha_depth %s " % max_alpha_depth
     29         print "pos.size %s " % self.pos.size
     30         print "dx.size %s " % self.dx.size
     31         print "dxlen.size %s " % self.dxlen.size
     32         print "color.size %s " % self.color.size


