rs-source(){   echo ${BASH_SOURCE} ; }
rs-edir(){ echo $(dirname $(rs-source)) ; }
rs-ecd(){  cd $(rs-edir); }

rs-repodir(){  echo $HOME/Rich_Simplified ; }
rs-dir(){      echo $(rs-repodir)/Rich_Simplified/src ; }

rs-rcd(){   cd $(rs-repodir); }
rs-cd(){   cd $(rs-dir); }

rs-vi(){   vi $(rs-source) ; }
rs-env(){  elocal- ; }
rs-usage(){ cat << EOU

git@github.com:simoncblyth/Rich_Simplified.git

Dear Simon,

During the recent meeting to prepare for the Hackathon, you mentioned about
getting a copy of the simple geometry that we hope to test during the
Hackathon. As you mentioned this would help in preparing for the Hackathon.

It is available at the following link, which you would be able to access:

https://gitlab.cern.ch/rs/opticks/-/tree/master/Rich_Simplified

In this page, the ‘include’ directory contains the header files, some of
which have the actual geometry parameters. The ‘src’ contains the code, which
is mostly similar to those in standard Geant4 examples. In particular the file
RichTbOpticksDetectorConstruction.cc has the Construct() method which creates
the full geometry. The RichTbGraphicsLbR.cc contains the graphics settings for
standard Geant4 visualization.

The main page of the link has a README file with some instructions on how to install this.

The geometry contains two spherical mirrors. One of them is ‘almost flat’
with a large radius of curvature. The geometry also has a quartz window and an
array of Mapmts arranged inside some structures. We also attach couple of
pictures of this geometry for illustration.

Hope this is sufficient information. If you have any questions related to this, please let us know.

BTW, thanks a lot for fixing issues related to conversion of spherical segments
into Opticks, as mentioned in your recent message to us on February 15. In our
case it was the ‘almost flat’ spherical mirror segment mentioned above, which
was causing the issues in this context. Whenever these fixes are ready and
available, please let us know so that we can try running Opticks with this
geometry.

Thanks and regards,

   Sajan, Adam, Keith, Yunlong, Lucas,Evelina and Marco. 



epsilon:~ blyth$ git clone https://gitlab.cern.ch/rs/opticks.git Rich_Simplified

Use Github web interface to creaty empty repo called "Rich_Simplified"

epsilon:Rich_Simplified blyth$ git remote add github git@github.com:simoncblyth/Rich_Simplified.git
epsilon:Rich_Simplified blyth$ git branch -M main

      -M
           Shortcut for --move --force.

      -m, --move
           Move/rename a branch and the corresponding reflog.

      -f, --force
           Reset <branchname> to <startpoint> if <branchname> exists already. Without -fgit branch refuses to change an existing branch. In combination with -d (or --delete), allow deleting the branch
           irrespective of its merged status. In combination with -m (or --move), allow renaming the branch even if the new branch name already exists.


epsilon:Rich_Simplified blyth$ git push -u github main

      -u, --set-upstream
           For every branch that is up to date or successfully pushed, add upstream (tracking) reference, used by argument-less git-pull(1) and other commands. For more information, see
           branch.<name>.merge in git-config(1).



Compare and copy over meaningful changes

epsilon:~ blyth$ diff -r -- brief rs Rich_Simplified
diff: extra operand `Rich_Simplified'
diff: Try `diff --help' for more information.
epsilon:~ blyth$ diff -r --brief rs Rich_Simplified
Files rs/.git/HEAD and Rich_Simplified/.git/HEAD differ
Files rs/.git/config and Rich_Simplified/.git/config differ
Files rs/.git/index and Rich_Simplified/.git/index differ
Files rs/.git/logs/HEAD and Rich_Simplified/.git/logs/HEAD differ
Only in Rich_Simplified/.git/logs/refs/heads: main
Only in rs/.git/logs/refs/heads: master
Only in Rich_Simplified/.git/logs/refs/remotes: github
Files rs/.git/logs/refs/remotes/origin/HEAD and Rich_Simplified/.git/logs/refs/remotes/origin/HEAD differ
Only in Rich_Simplified/.git/refs/heads: main
Only in rs/.git/refs/heads: master
Only in Rich_Simplified/.git/refs/remotes: github


Files rs/Rich_Simplified/CMakeLists.txt and Rich_Simplified/Rich_Simplified/CMakeLists.txt differ

    cp rs/Rich_Simplified/CMakeLists.txt Rich_Simplified/Rich_Simplified/CMakeLists.txt

Files rs/Rich_Simplified/Rich_Simplified.cc and Rich_Simplified/Rich_Simplified/Rich_Simplified.cc differ

     cp rs/Rich_Simplified/Rich_Simplified.cc Rich_Simplified/Rich_Simplified/Rich_Simplified.cc 

Files rs/Rich_Simplified/TimeTest.cc and Rich_Simplified/Rich_Simplified/TimeTest.cc differ

     cp rs/Rich_Simplified/TimeTest.cc  Rich_Simplified/Rich_Simplified/TimeTest.cc

Only in rs/Rich_Simplified: build
Files rs/Rich_Simplified/include/RichTbSimH.hh and Rich_Simplified/Rich_Simplified/include/RichTbSimH.hh differ


    diff rs/Rich_Simplified/include/RichTbSimH.hh Rich_Simplified/Rich_Simplified/include/RichTbSimH.hh 

Files rs/Rich_Simplified/src/EventAction.cc and Rich_Simplified/Rich_Simplified/src/EventAction.cc differ

    diff rs/Rich_Simplified/src/EventAction.cc Rich_Simplified/Rich_Simplified/src/EventAction.cc  


Files rs/Rich_Simplified/src/RichTbLHCbR1FlatMirror.cc and Rich_Simplified/Rich_Simplified/src/RichTbLHCbR1FlatMirror.cc differ

    diff rs/Rich_Simplified/src/RichTbLHCbR1FlatMirror.cc Rich_Simplified/Rich_Simplified/src/RichTbLHCbR1FlatMirror.cc 

Files rs/Rich_Simplified/src/RichTbLHCbR1SphMirror.cc and Rich_Simplified/Rich_Simplified/src/RichTbLHCbR1SphMirror.cc differ

    diff rs/Rich_Simplified/src/RichTbLHCbR1SphMirror.cc Rich_Simplified/Rich_Simplified/src/RichTbLHCbR1SphMirror.cc 


Files rs/Rich_Simplified/src/RichTbOpticksDetectorConstruction.cc and Rich_Simplified/Rich_Simplified/src/RichTbOpticksDetectorConstruction.cc differ

   diff rs/Rich_Simplified/src/RichTbOpticksDetectorConstruction.cc Rich_Simplified/Rich_Simplified/src/RichTbOpticksDetectorConstruction.cc 


Files rs/Rich_Simplified/src/RichTbSimH.cc and Rich_Simplified/Rich_Simplified/src/RichTbSimH.cc differ

   diff rs/Rich_Simplified/src/RichTbSimH.cc Rich_Simplified/Rich_Simplified/src/RichTbSimH.cc 


Only in Rich_Simplified/Rich_Simplified/src: RichTbVisManager.cc
Only in rs/Rich_Simplified/src: RichTbVisManager.cc_

   

Files rs/Rich_Simplified/src/SensitiveDetector.cc and Rich_Simplified/Rich_Simplified/src/SensitiveDetector.cc differ

   diff rs/Rich_Simplified/src/SensitiveDetector.cc Rich_Simplified/Rich_Simplified/src/SensitiveDetector.cc 

Only in rs/Rich_Simplified/src: load.sh
Only in rs/Rich_Simplified/src: tt.sh
epsilon:~ blyth$ 
epsilon:~ blyth$ 



2022-02-26 01:46:30.093 ERROR [402466] [main@59] ] load foundry 
2022-02-26 01:46:30.094 INFO  [402466] [main@63] CSGFoundry saved to cfbase /home/blyth/.opticks/geocache/TimeTest_WorldPhys_g4live/g4ok_gltf/a00339a20dfb266dc9d18565c704e41c/1/CSG_GGeo
2022-02-26 01:46:30.094 INFO  [402466] [main@64] logs are written to logdir /home/blyth/.opticks/geocache/TimeTest_WorldPhys_g4live/g4ok_gltf/a00339a20dfb266dc9d18565c704e41c/1/CSG_GGeo/logs
N[blyth@localhost CSG_GGeo]$ 
N[blyth@localhost CSG_GGeo]$ l /home/blyth/.opticks/geocache/TimeTest_WorldPhys_g4live/g4ok_gltf/a00339a20dfb266dc9d18565c704e41c/1/CSG_GGeo/
total 4


N[blyth@localhost CSGOptiX]$ l  /home/blyth/.opticks/geocache/TimeTest_WorldPhys_g4live/g4ok_gltf/a00339a20dfb266dc9d18565c704e41c/1/CSG_GGeo/CSGFoundry/


N[blyth@localhost CSGOptiX]$ cat /home/blyth/.opticks/geocache/TimeTest_WorldPhys_g4live/g4ok_gltf/a00339a20dfb266dc9d18565c704e41c/1/CSG_GGeo/CSGFoundry/mmlabel.txt
339:WorldBox
7:R1PmtMasterBox

MOI=RichTbR1SphRSphBox ./cxr_view.sh 

MOI=RichTbR1SphRSphBox EYE=-0.5,-0.5,-0.5 ./cxr_view.sh 
    not centralized : does that mean bbox issue ?

 MOI=RichTbR1SphRSphBox EYE=-1,0,0 ./cxr_view.sh 
     top edge of frame

MOI=RichTbR1SphRSphBox EYE=-1,0,0 LOOK=0,1,0 ./cxr_view.sh
     
MOI=RichTbR1SphRSphBox EYE=-1,0,0 LOOK=0,0,1 ./cxr_view.sh 
     near center of frame

MOI=RichTbR1SphRSphBox EYE=1,1,1 LOOK=0,0,1 ./cxr_view.sh 
    box near center

MOI=RichTbR1SphRSphBox EYE=-1,0,0 LOOK=0,0,1 ZOOM=3 ./cxr_view.sh 
    better view 

MOI=RichTbR1SphRSphBox EYE=-1,-1,1 LOOK=0,0,1 ZOOM=3 TMIN=1.6 ./cxr_view.sh 
    manages to cut into the pmtbox


N[blyth@localhost CSGOptiX]$ cat /home/blyth/.opticks/geocache/TimeTest_WorldPhys_g4live/g4ok_gltf/a00339a20dfb266dc9d18565c704e41c/1/CSG_GGeo/CSGFoundry/meshname.txt 
RichTbR1SphRSphBox_CSG_EXBB
RichTbR1FlatFull_CSG_EXBB
RichTbR1QwBox
R1ModuleBackPlBox
R1PmtAnodeBox
R1PmtQuartzBox
R1PmtPhCathBox
R1PmtFrontRingBox
R1PmtSideEnvBox
R1PmtSubMasterBox
R1PmtMasterBox
R1ECBox
R1ECBox
R1ECBox


MOI=R1PmtMasterBox ./cxr_view.sh 
    tight onto one of the small boxes

MOI=R1PmtMasterBox EYE=10,10,10 ./cxr_view.sh 

MOI=R1PmtMasterBox EYE=10,0,0 ./cxr_view.sh 
    bit perplexing view : must be edge onto the boxes

MOI=R1PmtMasterBox EYE=0,10,0 ./cxr_view.sh
    field of boxes to upper right 

MOI=R1PmtMasterBox EYE=0,50,0 ./cxr_view.sh 
    almost full frame green box only : must be outside

MOI=R1PmtMasterBox EYE=0,40,0 ./cxr_view.sh 
    still full frame of green

MOI=R1PmtMasterBox EYE=10,20,0 ./cxr_view.sh 
    good whacky angle view of the PMT boxes
    DONE

MOI=R1PmtMasterBox EYE=10,20,10 ./cxr_view.sh 
    more normal angle view of PMT boxes

EYE=1,0,0 MOI=WorldBox CAM=1 ./cxr_view.sh 
    curious black band

EYE=1,0,0 LOOK=0,0.1,0.1 MOI=WorldBox CAM=0 ZOOM=10 ./cxr_view.sh 
    good side view

EYE=1,0,0 LOOK=0,0.1,0.1 MOI=WorldBox CAM=1 ZOOM=10 TMIN=1 ./cxr_view.sh 
    manages to cut into the box so can see mirrors and detector

EYE=1,0,0 LOOK=0,0.1,0.1 MOI=WorldBox CAM=1 ZOOM=10 TMIN=0.8 ./cxr_view.sh 
    no cutting at 0.8

EYE=1,0,0 LOOK=0,0.1,0.1 MOI=WorldBox CAM=1 ZOOM=10 TMIN=0.95 ./cxr_view.sh 
    partial cut at 0.95

EYE=1,0,0 LOOK=0,0.1,0.1 MOI=WorldBox CAM=1 ZOOM=10 TMIN=0.96 ./cxr_view.sh 
    better at 0.96 
    DONE

EYE=1,0,0 MOI=RichTbR1MagShBox ./cxr_view.sh 
    good side view of detector elements
    DONE

EYE=0,1,0 MOI=RichTbR1MagShBox ./cxr_view.sh 
    good front on view of detector elements with circular tmin cut 

 EYE=0,1,0 MOI=RichTbR1MagShBox TMIN=0.5 ./cxr_view.sh 
    nice : bigger tmin cut circle 
    DONE

EYE=0,1,0 MOI=RichTbR1MagShBox TMIN=0 ./cxr_view.sh 
    green frame for all : TMIN 0, 0.1 

EYE=0,10,0 MOI=RichTbR1MagShBox  ./cxr_view.sh 
    green box and vertical line : is that the flat mirror or an artifact ?

EYE=-1,10,0 MOI=RichTbR1MagShBox  ./cxr_view.sh 
    
EYE=0,-10,0 MOI=RichTbR1MagShBox  ./cxr_view.sh 
    vertical line artifact 

EYE=0,-1,0 MOI=RichTbR1MagShBox  TMIN=0.5 ./cxr_view.sh 
    better back view
    DONE

EYE=0,-2.2,0 MOI=RichTbR1MagShBox   ./cxr_view.sh 
    see two mirrs and box
    DONE

EYE=0,-3,0 MOI=RichTbR1MagShBox   ./cxr_view.sh 
    too far away


EYE=1,-2.2,0 MOI=RichTbR1MagShBox   ./cxr_view.sh 



MOI=RichTbR1MagShBox ./cxr_view.sh 


N[blyth@localhost CSGOptiX]$ CSGTargetTest 
2022-02-28 00:55:25.218 INFO  [30352] [CSGTargetTest::CSGTargetTest@56] foundry CSGFoundry  total solids 2 STANDARD 2 ONE_PRIM 0 ONE_NODE 0 DEEP_COPY 0 KLUDGE_BBOX 0 num_prim 346 num_node 364 num_plan 0 num_tran 12 num_itra 12 num_inst 1057 ins 0 gas 0 ias 0 meshname 346 mmlabel 2
2022-02-28 00:55:25.219 INFO  [30352] [CSGTargetTest::dumpALL@118]  fd.getNumPrim 346 fd.meshname.size 346
 primIdx    0 lce (        0.00       0.00       0.00   15000.00 ) lce.w/1000        15.00 meshIdx  345 WorldBox
 primIdx    1 lce (        0.00       0.00       0.00    2600.00 ) lce.w/1000         2.60 meshIdx  344 RichTbR1MasterBox
 primIdx    2 lce (        0.00       0.00       0.00    2500.00 ) lce.w/1000         2.50 meshIdx  343 RichTbR1SubMasterBox
 primIdx    3 lce (        0.00     932.90   -1493.29    4159.82 ) lce.w/1000         4.16 meshIdx    0 RichTbR1SphRSphBox_CSG_EXBB
 primIdx    4 lce (        0.00 1269538.00 4837561.00 6105123.00 ) lce.w/1000      6105.12 meshIdx    1 RichTbR1FlatFull_CSG_EXBB
 primIdx    5 lce (        0.00    1465.00    1610.00     900.00 ) lce.w/1000         0.90 meshIdx  342 RichTbR1MagShBox
 primIdx    6 lce (        0.00       0.00       0.00     420.00 ) lce.w/1000         0.42 meshIdx    2 RichTbR1QwBox
 primIdx    7 lce (        0.00       0.00       0.00     550.00 ) lce.w/1000         0.55 meshIdx  341 R1PhDetSupFrameBox
 primIdx    8 lce (        0.00       0.00       0.00     310.00 ) lce.w/1000         0.31 meshIdx    3 R1ModuleBackPlBox
 primIdx    9 lce (        0.00       0.00       0.00      30.00 ) lce.w/1000         0.03 meshIdx   15 R1ModuleBox
 primIdx   10 lce (        0.00       0.00       0.00      30.00 ) lce.w/1000         0.03 meshIdx   11 R1ECBox
 primIdx   11 lce (        0.00       0.00       0.00      30.00 ) lce.w/1000         0.03 meshIdx   12 R1ECBox
 primIdx   12 lce (        0.00       0.00       0.00      30.00 ) lce.w/1000         0.03 meshIdx   13 R1ECBox
 primIdx   13 lce (        0.00       0.00       0.00      30.00 ) lce.w/1000         0.03 meshIdx   14 R1ECBox
 primIdx   14 lce (        0.00       0.00       0.00      30.00 ) lce.w/1000         0.03 meshIdx   20 R1ModuleBox
 primIdx   15 lce (        0.00       0.00       0.00      30.00 ) lce.w/1000         0.03 meshIdx   16 R1ECBox
 primIdx   16 lce (        0.00       0.00       0.00      30.00 ) lce.w/1000         0.03 meshIdx   17 R1ECBox










GGeo::save GGeoLib numMergedMesh 2 ptr 0x7fe816340df0
mm index   0 geocode   T                  numVolumes        339 numFaces        4116 numITransforms           1 numITransforms*numVolumes         339 GParts N GPts Y
mm index   1 geocode   T                  numVolumes          7 numFaces         120 numITransforms        1056 numITransforms*numVolumes        7392 GParts N GPts Y
 num_remainder_volumes 339 num_instanced_volumes 7392 num_remainder_volumes + num_instanced_volumes 7731 num_total_faces 4236 num_total_faces_woi 130836 (woi:without instancing) 

2022-02-25 20:11:30.344 INFO  [3584782] [GMeshLib::addAltMeshes@133]  num_indices_with_alt 1
2022-02-25 20:11:30.344 INFO  [3584782] [GMeshLib::addAltMeshes@143]  index with alt 0
2022-02-25 20:11:30.344 INFO  [3584782] [GMeshLib::dump@279] addAltMeshes meshnames 347 meshes 347
 i   0 aidx   0 midx   0 name                                 RichTbR1SphRSphBox mesh  nv     30 nf     56
 i   1 aidx   1 midx   1 name                                RichTbR1FlatFullDEV mesh  nv      8 nf     12
 i   2 aidx   2 midx   2 name                                      RichTbR1QwBox mesh  nv      8 nf     12
 i   3 aidx   3 midx   3 name                                  R1ModuleBackPlBox mesh  nv      8 nf     12
 i   4 aidx   4 midx   4 name                                      R1PmtAnodeBox mesh  nv      8 nf     12
 i   5 aidx   5 midx   5 name                                     R1PmtQuartzBox mesh  nv      8 nf     12
 i   6 aidx   6 midx   6 name                                     R1PmtPhCathBox mesh  nv      8 nf     12
 i   7 aidx   7 midx   7 name                                  R1PmtFrontRingBox mesh  nv     16 nf     32
 i   8 aidx   8 midx   8 name                                    R1PmtSideEnvBox mesh  nv     16 nf     28
 i   9 aidx   9 midx   9 name                                  R1PmtSubMasterBox mesh  nv      8 nf     12
 i  10 aidx  10 midx  10 name                                     R1PmtMasterBox mesh  nv      8 nf     12
 i  11 aidx  11 midx  11 name                                            R1ECBox mesh  nv      8 nf     12
 i  12 aidx  11 midx  12 name                                            R1ECBox mesh  nv      8 nf     12
 i  13 aidx  11 midx  13 name                                            R1ECBox mesh  nv      8 nf     12
 i  14 aidx  11 midx  14 name                                            R1ECBox mesh  nv      8 nf     12
 i  15 aidx  15 midx  15 name                                        R1ModuleBox mesh  nv      8 nf     12
 i  16 aidx  11 midx  16 name                                            R1ECBox mesh  nv      8 nf     12
 i  17 aidx  11 midx  17 name                                            R1ECBox mesh  nv      8 nf     12
 i  18 aidx  11 midx  18 name                                            R1ECBox mesh  nv      8 nf     12
 i  19 aidx  11 midx  19 name                                            R1ECBox mesh  nv      8 nf     12
 i  20 aidx  15 midx  20 name                                        R1ModuleBox mesh  nv      8 nf     12
 ...
 i 330 aidx  15 midx 330 name                                        R1ModuleBox mesh  nv      8 nf     12
 i 331 aidx  11 midx 331 name                                            R1ECBox mesh  nv      8 nf     12
 i 332 aidx  11 midx 332 name                                            R1ECBox mesh  nv      8 nf     12
 i 333 aidx  11 midx 333 name                                            R1ECBox mesh  nv      8 nf     12
 i 334 aidx  11 midx 334 name                                            R1ECBox mesh  nv      8 nf     12
 i 335 aidx  15 midx 335 name                                        R1ModuleBox mesh  nv      8 nf     12
 i 336 aidx  11 midx 336 name                                            R1ECBox mesh  nv      8 nf     12
 i 337 aidx  11 midx 337 name                                            R1ECBox mesh  nv      8 nf     12
 i 338 aidx  11 midx 338 name                                            R1ECBox mesh  nv      8 nf     12
 i 339 aidx  11 midx 339 name                                            R1ECBox mesh  nv      8 nf     12
 i 340 aidx  15 midx 340 name                                        R1ModuleBox mesh  nv      8 nf     12
 i 341 aidx 341 midx 341 name                                 R1PhDetSupFrameBox mesh  nv      8 nf     12
 i 342 aidx 342 midx 342 name                                   RichTbR1MagShBox mesh  nv     10 nf     16
 i 343 aidx 343 midx 343 name                               RichTbR1SubMasterBox mesh  nv      8 nf     12
 i 344 aidx 344 midx 344 name                                  RichTbR1MasterBox mesh  nv      8 nf     12
 i 345 aidx 345 midx 345 name                                           WorldBox mesh  nv      8 nf     12


GGeo::reportMeshUsage
 meshIndex, nvert, nface, nodeCount, nodeCount*nvert, nodeCount*nface, meshName, nmm, mm[0] 
     0 ( v   30 f   56 ) :       1 :         30 :         56 :                                 RichTbR1SphRSphBox :  1 :    0
     1 ( v    8 f   12 ) :       1 :          8 :         12 :                                RichTbR1FlatFullDEV :  1 :    0
     2 ( v    8 f   12 ) :       1 :          8 :         12 :                                      RichTbR1QwBox :  1 :    0
     3 ( v    8 f   12 ) :       1 :          8 :         12 :                                  R1ModuleBackPlBox :  1 :    0
     4 ( v    8 f   12 ) :    1056 :       8448 :      12672 :                                      R1PmtAnodeBox :  1 :    1
     5 ( v    8 f   12 ) :    1056 :       8448 :      12672 :                                     R1PmtQuartzBox :  1 :    1
     6 ( v    8 f   12 ) :    1056 :       8448 :      12672 :                                     R1PmtPhCathBox :  1 :    1
     7 ( v   16 f   32 ) :    1056 :      16896 :      33792 :                                  R1PmtFrontRingBox :  1 :    1
     8 ( v   16 f   28 ) :    1056 :      16896 :      29568 :                                    R1PmtSideEnvBox :  1 :    1
     9 ( v    8 f   12 ) :    1056 :       8448 :      12672 :                                  R1PmtSubMasterBox :  1 :    1
    10 ( v    8 f   12 ) :    1056 :       8448 :      12672 :                                     R1PmtMasterBox :  1 :    1
    11 ( v    8 f   12 ) :       1 :          8 :         12 :                                            R1ECBox :  1 :    0
    12 ( v    8 f   12 ) :       1 :          8 :         12 :                                            R1ECBox :  1 :    0
    13 ( v    8 f   12 ) :       1 :          8 :         12 :                                            R1ECBox :  1 :    0
    14 ( v    8 f   12 ) :       1 :          8 :         12 :                                            R1ECBox :  1 :    0
    15 ( v    8 f   12 ) :       1 :          8 :         12 :                                        R1ModuleBox :  1 :    0
    16 ( v    8 f   12 ) :       1 :          8 :         12 :                                            R1ECBox :  1 :    0


  336 ( v    8 f   12 ) :       1 :          8 :         12 :                                            R1ECBox :  1 :    0
   337 ( v    8 f   12 ) :       1 :          8 :         12 :                                            R1ECBox :  1 :    0
   338 ( v    8 f   12 ) :       1 :          8 :         12 :                                            R1ECBox :  1 :    0
   339 ( v    8 f   12 ) :       1 :          8 :         12 :                                            R1ECBox :  1 :    0
   340 ( v    8 f   12 ) :       1 :          8 :         12 :                                        R1ModuleBox :  1 :    0
   341 ( v    8 f   12 ) :       1 :          8 :         12 :                                 R1PhDetSupFrameBox :  1 :    0
   342 ( v   10 f   16 ) :       1 :         10 :         16 :                                   RichTbR1MagShBox :  1 :    0
   343 ( v    8 f   12 ) :       1 :          8 :         12 :                               RichTbR1SubMasterBox :  1 :    0
   344 ( v    8 f   12 ) :       1 :          8 :         12 :                                  RichTbR1MasterBox :  1 :    0
   345 ( v    8 f   12 ) :       1 :          8 :         12 :                                           WorldBox :  1 :    0
 tot  node :    7731 vert :   78768 face :  130836


2022-02-25 20:11:32.717 INFO  [3584782] [OGeo::convert@312] 
OGeo::convert GGeoLib numMergedMesh 2 ptr 0x7fe816340df0
mm index   0 geocode   T                  numVolumes        339 numFaces        4116 numITransforms           1 numITransforms*numVolumes         339 GParts Y GPts Y
mm index   1 geocode   T                  numVolumes          7 numFaces         120 numITransforms        1056 numITransforms*numVolumes        7392 GParts Y GPts Y
 num_remainder_volumes 339 num_instanced_volumes 7392 num_remainder_volumes + num_instanced_volumes 7731 num_total_faces 4236 num_total_faces_woi 130836 (woi:without instancing) 


Can get OTracerTest to view geom : so its there 




EOU
}


rs-get(){
   local dir=$(dirname $(rs-repodir)) &&  mkdir -p $dir && cd $dir

   git clone git@github.com:simoncblyth/Rich_Simplified.git 
}



