#!/usr/bin/env python
"""
TODO

* check the matrix transforms, by comparison against VRML2 output 


OBSERVATIONS

Russian doll Containment expectation is broken at the end, suggesting siblings being 
treated as ancestors. Probably some id stomping is happening::

    simon:pycollada blyth$ ./checkmonkey.py 
    INFO:monkey_collada:MonkeyCollada start normal loading 
    INFO:monkey_collada:_loadBoundGeometries starting
    INFO:monkey_collada:_loadBoundGeometries loaded 12230 
    <MBoundGeometry id=_dd_Geometry_PMT_lvMountRib20xb42d678 geom=pmt-rib-20xb3dceb0, 1 primitives, node depth 19>
    0 <MBoundGeometry id=World0xb5b2048 geom=WorldBox0xb3e6f60, 1 primitives, node depth 3>
    <BoundPolygons length=6>
    vtxmax [ 2400000.  2400000.  2400000.]
    vtxmin [-2400000. -2400000. -2400000.]
    1 <MBoundGeometry id=_dd_Geometry_Sites_lvNearSiteRock0xb5b1f08 geom=near_rock0xb3e6e30, 1 primitives, node depth 5>
    <BoundPolygons length=11>
    vtxmax [  18049.8515625 -767540.125       22890.       ]
    vtxmin [ -51089.8515625  -836679.875       -15103.79980469]
    2 <MBoundGeometry id=_dd_Geometry_Sites_lvNearHallBot0xb5b1618 geom=near_hall_bot0xb3e6ad8, 1 primitives, node depth 7>
    <BoundPolygons length=6>
    vtxmax [  -7561.66992188 -792262.3125       -2110.        ]
    vtxmin [ -25478.33007812 -811957.6875      -12410.        ]
    3 <MBoundGeometry id=_dd_Geometry_Pool_lvNearPoolDead0xb5b0940 geom=near_pool_dead_box0xb3e5f48, 1 primitives, node depth 9>
    <BoundPolygons length=96>
    vtxmax [  -9608.51757812 -794309.0625       -2110.        ]
    vtxmin [ -23431.78710938 -809910.8125      -12110.        ]
    4 <MBoundGeometry id=_dd_Geometry_Pool_lvNearPoolLiner0xb5afec8 geom=near_pool_liner_box0xb3e55f8, 1 primitives, node depth 11>
    <BoundPolygons length=64>
    vtxmax [  -9697.6875 -794398.375    -2110.    ]
    vtxmin [ -23342.05664062 -809821.75        -12026.        ]
    5 <MBoundGeometry id=_dd_Geometry_Pool_lvNearPoolOWS0xb4af5b0 geom=near_pool_ows_box0xb3e4c68, 1 primitives, node depth 13>
    <BoundPolygons length=152>
    vtxmax [  -9697.94335938 -794398.5          -2110.        ]
    vtxmin [ -23342.35546875 -809821.375       -12022.        ]
    6 <MBoundGeometry id=_dd_Geometry_Pool_lvNearPoolCurtain0xb4adfd0 geom=near_pool_curtain_box0xb3e06b0, 1 primitives, node depth 15>
    <BoundPolygons length=96>
    vtxmax [ -10766.80566406 -795467.375        -2110.        ]
    vtxmin [ -22273.46484375 -808752.5         -11022.        ]
    7 <MBoundGeometry id=_dd_Geometry_Pool_lvNearPoolIWS0xb42e398 geom=near_pool_iws_box0xb3dfce0, 1 primitives, node depth 17>
    <BoundPolygons length=102>
    vtxmax [ -10766.578125 -795467.25       -2110.      ]
    vtxmin [ -22273.1953125 -808752.8125     -11018.       ]
    8 <MBoundGeometry id=_dd_Geometry_PMT_lvMountRib20xb42d678 geom=pmt-rib-20xb3dceb0, 1 primitives, node depth 15>
    <BoundPolygons length=6>
    vtxmax [ -21818.61914062 -806152.4375       -3006.96777344]
    vtxmin [ -21966.02148438 -806203.9375       -3122.23242188]
    9 <MBoundGeometry id=_dd_Geometry_PMT_lvMountRib20xb42d678 geom=pmt-rib-20xb3dceb0, 1 primitives, node depth 19>
    <BoundPolygons length=6>
    vtxmax [ -17993.8671875  -808480.6875       -5312.34423828]
    vtxmin [ -18134.58984375 -808542.5625       -5421.45556641]
    simon:pycollada blyth$ 



"""
import logging, random, sys
log = logging.getLogger(__name__)
from monkey_collada import MonkeyCollada as Collada


def examine(dae, ibg=None):
    if ibg is None:
        ibg = random.randint(0,len(dae.bound_geometries)-1)

    bg = dae.bound_geometries[ibg]
    log.info("ibg %s %s " % (ibg, bg))    
    print bg 

    bgs = [dae.bound_geometries[n] for n in filter(lambda _:_.__class__.__name__ =='MonkeyNodeNode',bg.path)] + [bg] 

    for i,g in enumerate(bgs): 
        print i, g
        for p in g.primitives():
            print p
            #tris = p.triangleset()
            #print tris
            print "vtxmax", p.vertex.max(axis=0)
            print "vtxmin", p.vertex.min(axis=0)
            print "vtxdif", p.vertex.max(axis=0)-p.vertex.min(axis=0)



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv)>1:
        arg = int(sys.argv[1])
    else: 
        arg = None

    dae = Collada("test.dae")
    examine(dae, arg)





