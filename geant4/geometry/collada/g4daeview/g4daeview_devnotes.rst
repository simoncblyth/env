Daeview Dev Notes 
=====================


Limitations
-------------

GPU Out-of-memory during BVH construction with full Juno geometry (50M nodes?)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Restricting to about half the geometry succeeds::

     g4daeview.sh -p juno -g 1:25000 --with-chroma --launch 3,3,1


Maybe can reorganize the work to avoid using too much memory ?::

    g4daeview.sh -p juno --with-chroma

    074 2014-05-26 10:51:22,642 env.geant4.geometry.collada.collada_to_chroma:297 ColladaToChroma adding BVH
    075 2014-05-26 10:51:23,879 chroma.loader       :155 Building new BVH using recursive grid algorithm.
    076 Expanding 422925 parent nodes
    077 Merging 50232688 nodes to 15826587 parents
    078 Expanding 48194 parent nodes
    079 Merging 16250194 nodes to 4923755 parents
    080 Merging 4971964 nodes to 1289438 parents
    081 Merging 1289438 nodes to 266462 parents
    082 Merging 266462 nodes to 51806 parents
    083 Merging 51806 nodes to 10332 parents
    084 Merging 10332 nodes to 2216 parents
    085 Merging 2216 nodes to 480 parents
    086 Merging 480 nodes to 104 parents
    087 Merging 104 nodes to 32 parents
    088 Merging 32 nodes to 8 parents
    089 Merging 8 nodes to 2 parents
    090 Merging 2 nodes to 1 parent
    091 Traceback (most recent call last):
    092   File "/Users/blyth/env/bin/g4daeview.py", line 4, in <module>
    093     main()
    094   File "/usr/local/env/chroma_env/lib/python2.7/site-packages/env/geant4/geometry/collada/g4daeview/g4daeview.py", line 186, in main
    095     scene = DAEScene(geometry, config )
    096   File "/usr/local/env/chroma_env/lib/python2.7/site-packages/env/geant4/geometry/collada/g4daeview/daescene.py", line 45, in __init__
    097     chroma_geometry = geometry.make_chroma_geometry()
    098   File "/usr/local/env/chroma_env/lib/python2.7/site-packages/env/geant4/geometry/collada/g4daeview/daegeometry.py", line 315, in make_chroma_ge    ometry
    099     cc.convert_geometry(nodes=self.nodes())
    100   File "/usr/local/env/chroma_env/lib/python2.7/site-packages/env/geant4/geometry/collada/collada_to_chroma.py", line 291, in convert_geometry
    101     self.add_bvh()
    102   File "/usr/local/env/chroma_env/lib/python2.7/site-packages/env/geant4/geometry/collada/collada_to_chroma.py", line 304, in add_bvh
    103     cuda_device=cuda_device)
    104   File "/usr/local/env/chroma_env/src/chroma/chroma/loader.py", line 160, in load_bvh
    105     bvh = make_recursive_grid_bvh(geometry.mesh, target_degree=3)
    106   File "/usr/local/env/chroma_env/src/chroma/chroma/bvh/grid.py", line 91, in make_recursive_grid_bvh
    107     nodes, layer_bounds = concatenate_layers(layers)
    108   File "/usr/local/env/chroma_env/src/chroma/chroma/gpu/bvh.py", line 266, in concatenate_layers
    109     grid=(nblocks_this_iter,1))
    110   File "/usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/driver.py", line 355, in function_call
    111     handlers, arg_buf = _build_arg_buf(args)
    112   File "/usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/driver.py", line 125, in _build_arg_buf
    113     arg_data.append(int(arg.get_device_alloc()))
    114   File "/usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/driver.py", line 59, in get_device_alloc
    115     self.dev_alloc = mem_alloc_like(self.array)
    116   File "/usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/driver.py", line 608, in mem_alloc_like
    117     return mem_alloc(ary.nbytes)
    118 pycuda._driver.MemoryError: cuMemAlloc failed: out of memory






Old Issues To Be Revisited
---------------------------

More Bizarrness
~~~~~~~~~~~~~~~~

Off into outerspace and back::

    (chroma_env)delta:daeview blyth$ grep ESR solids.txt
    1274  DAESolid v 578  t 1188  n 578   : 4427 __dd__Geometry__AdDetails__lvTopRefGap--pvTopESR0xab4bd50.0   
    1277  DAESolid v 330  t 688  n 330   : 4430 __dd__Geometry__AdDetails__lvBotRefGap--pvBotESR0xae4eda0.0   
    2934  DAESolid v 578  t 1188  n 578   : 6087 __dd__Geometry__AdDetails__lvTopRefGap--pvTopESR0xab4bd50.1   
    2937  DAESolid v 330  t 688  n 330   : 6090 __dd__Geometry__AdDetails__lvBotRefGap--pvBotESR0xae4eda0.1   
    (chroma_env)delta:daeview blyth$ 
    (chroma_env)delta:daeview blyth$ ./udp.py "-j 4427_-5,2,2:"
    sending -j 4427_-5,2,2: 

Its because its a 4.5 disk shape, look from above::

    udp.py "-t 4427_0,0.001,-2"
    udp.py "-t 4427_0,0.001,5"

    ./udp.py "-t 4427_0,0.001,-5"   # view upwards from bottom of AD 


Animation
~~~~~~~~~~~

::

    (chroma_env)delta:daeview blyth$ ./daegeometry.py > solids.txt
    INFO:env.geant4.geometry.collada.daenode:DAENode.parse pycollada parse /Users/blyth/env/geant4/geometry/materials/g4_00.dae 
    INFO:__main__:Flattening 9077 DAESolid into one DAEMesh...
    INFO:__main__:DAEMesh v 1234834  t 2438640  n 1234834 
    (chroma_env)delta:daeview blyth$ vi solids.txt 
    (chroma_env)delta:daeview blyth$ grep ball solids.txt
    (chroma_env)delta:daeview blyth$ grep Ball solids.txt
    1369  DAESolid v 267  t 528  n 267   : 4522 __dd__Geometry__CalibrationSources__lvWallLedSourceAssy--pvWallLedDiffuserBall0xab71f78.0   
    1372  DAESolid v 267  t 528  n 267   : 4525 __dd__Geometry__CalibrationSources__lvWallLedSourceAssy--pvWallLedDiffuserBall0xab71f78.1   
    1375  DAESolid v 267  t 528  n 267   : 4528 __dd__Geometry__CalibrationSources__lvWallLedSourceAssy--pvWallLedDiffuserBall0xab71f78.2   
    1389  DAESolid v 267  t 528  n 267   : 4542 __dd__Geometry__CalibrationSources__lvLedSourceShell--pvDiffuserBall0xabe00c8.0   
    1477  DAESolid v 267  t 528  n 267   : 4630 __dd__Geometry__CalibrationSources__lvLedSourceShell--pvDiffuserBall0xabe00c8.1   
    1559  DAESolid v 267  t 528  n 267   : 4712 __dd__Geometry__CalibrationSources__lvLedSourceShell--pvDiffuserBall0xabe00c8.2   
    3029  DAESolid v 267  t 528  n 267   : 6182 __dd__Geometry__CalibrationSources__lvWallLedSourceAssy--pvWallLedDiffuserBall0xab71f78.3   
    3032  DAESolid v 267  t 528  n 267   : 6185 __dd__Geometry__CalibrationSources__lvWallLedSourceAssy--pvWallLedDiffuserBall0xab71f78.4   
    3035  DAESolid v 267  t 528  n 267   : 6188 __dd__Geometry__CalibrationSources__lvWallLedSourceAssy--pvWallLedDiffuserBall0xab71f78.5   
    3049  DAESolid v 267  t 528  n 267   : 6202 __dd__Geometry__CalibrationSources__lvLedSourceShell--pvDiffuserBall0xabe00c8.3   
    3137  DAESolid v 267  t 528  n 267   : 6290 __dd__Geometry__CalibrationSources__lvLedSourceShell--pvDiffuserBall0xabe00c8.4   
    3219  DAESolid v 267  t 528  n 267   : 6372 __dd__Geometry__CalibrationSources__lvLedSourceShell--pvDiffuserBall0xabe00c8.5   
    (chroma_env)delta:daeview blyth$ 
    (chroma_env)delta:daeview blyth$ g4daeview.py -t +0 -j +1369,+1372,+1375,+1389,+1477,+1559,+3029,+3032,+3035,+3049,+3137,+3219 --near 1e-5


Reflectors
------------

::

    (chroma_env)delta:daeview blyth$ grep TopReflector solids.txt
    1272  DAESolid v 296  t 608  n 296   : 4425 __dd__Geometry__AD__lvOIL--pvTopReflector0xab22490.0   
    1273  DAESolid v 296  t 608  n 296   : 4426 __dd__Geometry__AdDetails__lvTopReflector--pvTopRefGap0xabcc228.0   
    2932  DAESolid v 296  t 608  n 296   : 6085 __dd__Geometry__AD__lvOIL--pvTopReflector0xab22490.1   
    2933  DAESolid v 296  t 608  n 296   : 6086 __dd__Geometry__AdDetails__lvTopReflector--pvTopRefGap0xabcc228.1   
    (chroma_env)delta:daeview blyth$ 

    ./udp.py "-t 4425_0,0.001,-2_-2,-2,0
    # looking up at top reflector


Bizarre
~~~~~~~~

Issues when small extent ?

::

    g4daeview.py -t +1369

    g4daeview.py -t +1369 --eye=0,0.001,20    # small ball and cylinder



Interpolation
~~~~~~~~~~~~~~~~


Expected yoyo, just get fall::

    g4daeview.py -t 8153 --eye="2,2,40" --look="2,2.001,0" -j +0_2,2,-40:+0_2,2,40    



Very long shapes are problematic::

    g4daeview.py -t 4522 -j 4522_0,5,0:4522_5,0,0:4522_0,0.001,5 --near 1e-6 --far 1e6

    g4daeview.py -g 4522,4525,4528,4542,4630,4712

    g4daeview.py -g 4522,4525,4528,4542,4630,4712 -t "" -j 4522:4525:4528:4542:4630:4712

    g4daeview.py -g 4522:4712 -t 4522

    g4daeview.py -t 4522 -j 4522_0,5,0:4522_5,0,0:4522_0,0.001,5 --near 0.01

    g4daeview.py -t 4522 -j 4522_0,5,0:4522_5,0,0:4522_0,0.001,5 



Interpolation Jumps (FIXED)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


::

    g4daeview.py -t +1000 -j +1000_2,2,2:+1000_2,2,10

    g4daeview.py -t 4522 -j 4522_0,5,0:4522_5,0,0:4522_0,0.001,5



