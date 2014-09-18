#!/usr/bin/env python
"""
IDMAP
=======

Read .idmap file into numpy arrays and python dict 

This is used by daenode.py to idmaplink the channel_id into 
the daenode instances.

#. OK with partial geometry loading ?

   * its not the collada parse that can be partial 
     but rather the presentation of nodes from that full parse 


Cheat Placement
----------------

::

    delta:~ blyth$ echo $IDMAP
    /usr/local/env/geant4/geometry/export/DayaBay_MX_20140916-2050/g4_00.idmap
    delta:~ blyth$ cp $IDMAP /usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.idmap
    delta:~ blyth$ 


Testing idmap parse
--------------------

::

   export IDMAP=/usr/local/env/geant4/geometry/export/DayaBay_MX_20140916-2050/g4_00.idmap
   ./idmap.py $IDMAP

::

    # GiGaRunActionExport::WriteIdMap fields: index,pmtid,pmtid(hex),pvname  npv:12230
    0 0 0  Universe
    1 0 0  /dd/Structure/Sites/db-rock
    2 0 0  /dd/Geometry/Sites/lvNearSiteRock#pvNearHallTop
    3 0 0  /dd/Geometry/Sites/lvNearHallTop#pvNearTopCover
    4 0 0  /dd/Geometry/Sites/lvNearHallTop#pvNearTeleRpc#pvNearTeleRpc:1
    5 0 0  /dd/Geometry/RPC/lvRPCMod#pvRPCFoam
    6 0 0  /dd/Geometry/RPC/lvRPCFoam#pvBarCham14Array#pvBarCham14ArrayOne:1#pvBarCham14Unit
    7 0 0  /dd/Geometry/RPC/lvRPCBarCham14#pvRPCGasgap14


Parsed Array
-------------

::


    In [102]: a[a['id'] != 0]
    Out[102]: 
    array([ (3199, 16843009, '1010101', '/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:1#pvAdPmtUnit#pvAdPmt'),
           (3200, 16843009, '1010101', '/dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum'),
           (3201, 16843009, '1010101', '/dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode'),
           ...,
           (11422, 17172488, '1060808', '/dd/Geometry/Pool/lvNearPoolOWS#pvVetoPmtNearOutFaceout#pvNearOutFaceoutWall8#pvNearOutFaceoutWall8:8#pvVetoPmtUnit#pvPmtMount#pvMountRib3s#pvMountRib3s:1#pvMountRib3unit'),
           (11423, 17172488, '1060808', '/dd/Geometry/Pool/lvNearPoolOWS#pvVetoPmtNearOutFaceout#pvNearOutFaceoutWall8#pvNearOutFaceoutWall8:8#pvVetoPmtUnit#pvPmtMount#pvMountRib3s#pvMountRib3s:2#pvMountRib3unit'),
           (11424, 17172488, '1060808', '/dd/Geometry/Pool/lvNearPoolOWS#pvVetoPmtNearOutFaceout#pvNearOutFaceoutWall8#pvNearOutFaceoutWall8:8#pvVetoPmtUnit#pvPmtMount#pvMountRib3s#pvMountRib3s:3#pvMountRib3unit')], 
          dtype=[('index', '<i4'), ('id', '<i4'), ('idhex', 'S7'), ('pvname', 'S256')])


"""
import os, sys, logging
import numpy as np

log = logging.getLogger(__name__)

class IDMap(dict):
    dtype = [
               ('index',np.int32),
               ('id',np.int32),
               ('idhex','|S7'), 
               ('pvname','|S256')
             ]
    def __init__(self, path):
        dict.__init__(self)
        # Cannot use default hash comment marker as that is meaningful in pvnames
        a = np.genfromtxt(path,comments=None,skip_header=1,dtype=self.dtype)
        assert np.all( np.arange(len(a),dtype=np.int32) == a['index'] )
        uid = np.unique(a['id'])
        log.info("found %s unique ids " % (len(uid)))
        log.debug("ids %s  " % (repr(uid)))
        self.a = a 
        self.update(dict(zip(a['index'],a['id'])))
        assert len(self) == len(a) 


def main():
    idmap = IDMap(sys.argv[1])


if __name__ == '__main__':
    #main()    
    logging.basicConfig(level=logging.INFO)
    path = os.environ['IDMAP']
    idmap = IDMap(path)




