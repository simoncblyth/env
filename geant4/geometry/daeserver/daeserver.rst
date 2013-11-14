DAESERVER
==========




gather pieces
---------------

::

    simon:WebGLBook blyth$ pwd
    /usr/local/env/graphics/webgl/WebGLBook

    simon:WebGLBook blyth$ cp -r sim ~/e/geant4/geometry/daeserver/static/
    simon:WebGLBook blyth$ cp -r css ~/e/geant4/geometry/daeserver/static/
    simon:WebGLBook blyth$ cp -r libs ~/e/geant4/geometry/daeserver/static/
    simon:WebGLBook blyth$ cp -r Chapter\ 7 ~/e/geant4/geometry/daeserver/static/   
    simon:WebGLBook blyth$ cp -r models/dybgeom ~/e/geant4/geometry/daeserver/static/models/ 

    http://localhost/geo/static/Chapter%207/production-loader-collada.html


::

    simon:static blyth$ cp Chapter\ 7/colladaModel.js .
    simon:static blyth$ cp Chapter\ 7/modelViewer.js .


ideas
------

* volume coloring to distinguish levels/materials
* cannot copy/paste from the text as it causes rotations

bits of geometry
------------------

* http://belle7.nuu.edu.tw/dae/tree/3166___0.html
 
  * with an hole 

* http://belle7.nuu.edu.tw/dae/tree/3160___0.html?wireframe=1

  * some little thing offset in its frame

* http://belle7.nuu.edu.tw/dae/tree/3159___0.html?wireframe=1

  * GdDopedLS

* http://belle7.nuu.edu.tw/dae/tree/3159___5.html?wireframe=1

  * any children ? nope

::

         .           [5.176] VNode(11,13)[3145,__dd__Geometry__RPCSupport__lvNearHbeamBigUnit--pvNearRightDiagSIRightY40xa8d34e0.3] __dd__Materials__Iron0x8839b40  
                     [5.177] VNode(11,13)[3146,__dd__Geometry__RPCSupport__lvNearHbeamBigUnit--pvNearRightDiagSILeftY40xa8d35c8.3] __dd__Materials__Iron0x8839b40  
         [2.1] VNode(5,7)[3147,__dd__Geometry__Sites__lvNearSiteRock--pvNearHallBot0xaa8b020.0] __dd__Materials__Rock0x8868188  
             [3.0] VNode(7,9)[3148,__dd__Geometry__Sites__lvNearHallBot--pvNearPoolDead0xaa8a0b0.0] __dd__Materials__DeadWater0x8867010  
                 [4.0] VNode(9,11)[3149,__dd__Geometry__Pool__lvNearPoolDead--pvNearPoolLiner0xaa89650.0] __dd__Materials__Tyvek0x8865cb8  
                     [5.0] VNode(11,13)[3150,__dd__Geometry__Pool__lvNearPoolLiner--pvNearPoolOWS0xaa88da8.0] __dd__Materials__OwsWater0x8866e08  
                         [6.0] VNode(13,15)[3151,__dd__Geometry__Pool__lvNearPoolOWS--pvNearPoolCurtain0xa9883f0.0] __dd__Materials__Tyvek0x8865cb8  
                             [7.0] VNode(15,17)[3152,__dd__Geometry__Pool__lvNearPoolCurtain--pvNearPoolIWS0xa986bf8.0] __dd__Materials__IwsWater0x8865388  
                                 [8.0] VNode(17,19)[3153,__dd__Geometry__Pool__lvNearPoolIWS--pvNearADE10xa9076d8.0] __dd__Materials__IwsWater0x8865388  
                                     [9.0] VNode(19,21)[3154,__dd__Geometry__AD__lvADE--pvSST0xa906040.0] __dd__Materials__StainlessSteel0x8887430  
                                         [10.0] VNode(21,23)[3155,__dd__Geometry__AD__lvSST--pvOIL0xa8ebb48.0] __dd__Materials__MineralOil0x8861428  
                                             [11.0] VNode(23,25)[3156,__dd__Geometry__AD__lvOIL--pvOAV0xa8d9328.0] __dd__Materials__Acrylic0x8880fd8  
                                                 [12.0] VNode(25,27)[3157,__dd__Geometry__AD__lvOAV--pvLSO0xa8d68e0.0] __dd__Materials__LiquidScintillator0x8882718  
                                                     [13.0] VNode(27,29)[3158,__dd__Geometry__AD__lvLSO--pvIAV0xa8d4990.0] __dd__Materials__Acrylic0x8880fd8  
                                                         [14.0] VNode(29,31)[3159,__dd__Geometry__AD__lvIAV--pvGDS0xa8d40f8.0] __dd__Materials__GdDopedLS0x8880cb0  
                                                         [14.1] VNode(29,31)[3160,__dd__Geometry__AD__lvIAV--pvOcrGdsInIAV0xa8d40a8.0] __dd__Materials__GdDopedLS0x8880cb0  
                                                     [13.1] VNode(27,29)[3161,__dd__Geometry__AD__lvLSO--pvIavTopHub0xa8d49c8.0] __dd__Materials__Acrylic0x8880fd8  
                                                     [13.2] VNode(27,29)[3162,__dd__Geometry__AD__lvLSO--pvCtrGdsOflBotClp0xa8d4a00.0] __dd__Materials__Acrylic0x8880fd8  
                                                     [13.3] VNode(27,29)[3163,__dd__Geometry__AD__lvLSO--pvCtrGdsOflTfbInLso0xa8d4ac8.0] __dd__Materials__Teflon0x8881280  
                                                     [13.4] VNode(27,29)[3164,__dd__Geometry__AD__lvLSO--pvCtrGdsOflInLso0xa8d4b48.0] __dd__Materials__GdDopedLS0x8880cb0  
                                                     [13.5] ...  
                                                     [13.31] VNode(27,29)[3191,__dd__Geometry__AD__lvLSO--pvIavTopRibs--IavRibs..4--IavTopRibRot0xa8d6140.0] __dd__Materials__Acrylic0x8880fd8  
                                                     [13.32] VNode(27,29)[3192,__dd__Geometry__AD__lvLSO--pvIavTopRibs--IavRibs..5--IavTopRibRot0xa8d6230.0] __dd__Materials__Acrylic0x8880fd8  
                                                     [13.33] VNode(27,29)[3193,__dd__Geometry__AD__lvLSO--pvIavTopRibs--IavRibs..6--IavTopRibRot0xa8d63c0.0] __dd__Materials__Acrylic0x8880fd8  
                                                     [13.34] VNode(27,29)[3194,__dd__Geometry__AD__lvLSO--pvIavTopRibs--IavRibs..7--IavTopRibRot0xa8d64b0.0] __dd__Materials__Acrylic0x8880fd8  
                                                 [12.1] VNode(25,27)[3195,__dd__Geometry__AD__lvOAV--pvOcrGdsLsoInOav0xa8d6890.0] __dd__Materials__LiquidScintillator0x8882718  
                                                     [13.0] VNode(27,29)[3196,__dd__Geometry__AdDetails__lvOcrGdsLsoInOav--pvOcrGdsTfbInOav0xa8d6678.0] __dd__Materials__Teflon0x8881280  
                                                         [14.0] VNode(29,31)[3197,__dd__Geometry__AdDetails__lvOcrGdsTfbInOav--pvOcrGdsInOav0xa8d6448.0] __dd__Materials__GdDopedLS0x8880cb0  
                                                 [12.2] VNode(25,27)[3198,__dd__Geometry__AD__lvOAV--pvOcrCalLsoInOav0xa8d6960.0] __dd__Materials__LiquidScintillator0x8882718  
                                             [11.1] VNode(23,25)[3199,__dd__Geometry__AD__lvOIL--pvAdPmtArray--pvAdPmtArrayRotated--pvAdPmtRingInCyl..1--pvAdPmtInRing..1--pvAdPmtUnit--pvAdPmt0xa8d92d8.0] __dd__Materials__P



* http://belle7.nuu.edu.tw/dae/tree/3157___2.html?wireframe=1

  * clearly children here, but too heavy 



