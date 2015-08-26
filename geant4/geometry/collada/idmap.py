#!/usr/bin/env python
"""
IDMAP
=======

::

    ipython idmap.py -i


Read .idmap file into numpy arrays and python dict 

This is used by daenode.py to idmaplink the channel_id into 
the daenode instances.

#. OK with partial geometry loading ?

   * its not the collada parse that can be partial 
     but rather the presentation of nodes from that full parse 

Creation of IDMAP
------------------

Contortions needed to get IdMap as it does not 
exist at Geant4 level, so must cross-reference from 
Gaudi/Gauss level DetectorElement to a 
Geant4 full geometry traverse.


GiGaRunActionExport::WriteIdMap
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* traverse volume tree, collecting pvStack for 
  each visited node 

* recreate a G4TouchableHistory from stacks

* use touchable to access the DetectorElement pmtid 
  collects that into a vector

* writes out the vectors into idmap file 


NuWa-trunk/lhcb/Sim/GaussTools/src/Components/GiGaRunActionExport.cpp::

    670           case 'M':
    671                  WriteIdMap( wpv, FreeFilePath(base, ".idmap"));
    672                  break;

    453 void GiGaRunActionExport::WriteIdMap(G4VPhysicalVolume* wpv, const G4String& path )
    454 {
    455    // collect identifiers from full traverse, 
    456    // many placeholder zeros expected
    457 
    458    std::cout << "GiGaRunActionExport::WriteIdMap to " << path
    459              << " WorldVolume : " << wpv->GetName()
    460              << std::endl ;
    461 
    462    const G4LogicalVolume* lvol = wpv->GetLogicalVolume();
    463 
    464    m_pvid.clear();
    465    m_pvname.clear();
    466 
    467    // manual World entry, for indice alignment 
    468    m_pvid.push_back(0);
    469    m_pvname.push_back(wpv->GetName());
    470 
    471    PVStack_t pvStack ;     // Universe not on stack
    472    TraverseVolumeTree( lvol, pvStack );
    473 
    474    assert( m_pvid.size() == m_pvname.size() );
    475    size_t npv = m_pvid.size() ;
    476 
    477    std::ofstream fp;
    478    fp.open(path);
    479 
    480    fp << "# GiGaRunActionExport::WriteIdMap fields: index,pmtid,pmtid(hex),pvname  npv:" << npv << '\n' ;
    481 
    482    for( size_t index=0; index < npv; ++index ){
    483        int id = m_pvid[index] ;
    484        std::string name = m_pvname[index] ;    // for debug, NOT identity matching 


idmap csv file
~~~~~~~~~~~~~~~


/usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.idmap::

    1 # GiGaRunActionExport::WriteIdMap fields: index,pmtid,pmtid(hex),pvname  npv:12230
    2 0 0 0  Universe
    3 1 0 0  /dd/Structure/Sites/db-rock
    4 2 0 0  /dd/Geometry/Sites/lvNearSiteRock#pvNearHallTop
    5 3 0 0  /dd/Geometry/Sites/lvNearHallTop#pvNearTopCover
    ...
    11343 11341 17172484 1060804  /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum
    11344 11342 17172484 1060804  /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode

* read .idmap with env/geant4/geometry/collada/idmap.py


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




Adding transform info (2014-10-13)
-------------------------------------

::

    delta:collada blyth$ head -10 /usr/local/env/geant4/geometry/export/DayaBay_MX_20141013-1711/g4_00.idmap 
    # GiGaRunActionExport::WriteIdMap fields: index,pmtid,pmtid(hex),pvname  npv:12230
    0 0 0 (0,0,0) (1,0,0)(0,1,0)(0,0,1) Universe
    1 0 0 (664494,-449556,2110) (-0.543174,-0.83962,0)(0.83962,-0.543174,0)(0,0,1) /dd/Structure/Sites/db-rock
    2 0 0 (661994,-449056,-5390) (-0.543174,-0.83962,0)(0.83962,-0.543174,0)(0,0,1) /dd/Geometry/Sites/lvNearSiteRock#pvNearHallTop
    3 0 0 (664494,-449556,2088) (-0.543174,-0.83962,0)(0.83962,-0.543174,0)(0,0,1) /dd/Geometry/Sites/lvNearHallTop#pvNearTopCover
    4 0 0 (668975,-437058,-683.904) (-0.53472,-0.84503,0)(0.84503,-0.53472,0)(0,0,1) /dd/Geometry/Sites/lvNearHallTop#pvNearTeleRpc#pvNearTeleRpc:1
    5 0 0 (668985,-437063,-683.904) (-0.53472,-0.84503,0)(0.84503,-0.53472,0)(0,0,1) /dd/Geometry/RPC/lvRPCMod#pvRPCFoam


::

    In [4]: rot[1000]
    Out[4]: 
    array([[-0.53929  , -0.841109 , -0.0412639],
           [ 0.84212  , -0.538642 , -0.0264252],
           [ 0.       , -0.049    ,  0.998799 ]])

    In [5]: tra[1000]
    Out[5]: array([ 662561. , -447524. ,  -20583.8])




::

    name=env/geant4/geometry/export/DayaBay_MX_20141013-1711/g4_00.idmap
    path=$(local-base)/$name
    mkdir -p $(dirname $path)
    scp N:$(local-base N)/$name $path 

    export IDMAP=$path



Cheat Placement Again
----------------------

::

    (chroma_env)delta:DayaBay_VGDX_20140414-1300 blyth$ mv g4_00.idmap g4_00.idmap.sep17
    (chroma_env)delta:DayaBay_VGDX_20140414-1300 blyth$ echo $IDMAP
    /usr/local/env/geant4/geometry/export/DayaBay_MX_20141013-1711/g4_00.idmap
    (chroma_env)delta:DayaBay_VGDX_20140414-1300 blyth$ cp $IDMAP .





"""
import os, sys, logging
import numpy as np

try:
    import IPython as IP
except ImportError:
    IP = None


log = logging.getLogger(__name__)

class IDMap(dict):
    old_dtype = [
               ('index',np.int32),
               ('id',np.int32),
               ('idhex','|S7'), 
               ('pvname','|S256'),
             ]
    dtype = [
               ('index',np.int32),
               ('id',np.int32),
               ('idhex','|S7'), 
               ('trans','|S256'),
               ('rotrow','|S256'),
               ('pvname','|S256'),
             ]


    def __init__(self, path=None, old=False):
        if path is None:
            path = os.environ['DAE_NAME_DYB_IDMAP'] # define with: export-;export-export
        pass 
        dict.__init__(self)
        # Cannot use default hash comment marker as that is meaningful in pvnames
        log.info("np.genfromtxt %s " % path ) 

        dtype = self.old_dtype if old else self.dtype
        a = np.genfromtxt(path,comments=None,skip_header=1,dtype=dtype) #, converters=dict(rotrow=lambda _:"yep%s"%_))
        assert np.all( np.arange(len(a),dtype=np.int32) == a['index'] )
        uid = np.unique(a['id'])
        log.info("found %s unique ids " % (len(uid)))
        log.debug("ids %s  " % (repr(uid)))
        self.a = a 
        self.update(dict(zip(a['index'],a['id'])))

        if not old:
            rot = np.zeros( (len(a),3,3) )
            tra = np.zeros( (len(a),3) )
            transform = np.zeros( (len(a),4,4))
       
            for i,rec in enumerate(a):
                tra[i] = np.fromstring(rec['trans'][1:-1],sep=",").reshape((3,))
                rot[i] = np.fromstring(rec['rotrow'][1:-1].replace(")(",","),sep=",").reshape((3,3))
                transform[i] = np.identity(4)
                transform[i][:3,:3] = rot[i]
                transform[i][:3,3] = tra[i]
            pass
            self.tra = tra
            self.rot = rot
            self.transform = transform
        pass

        assert len(self) == len(a) 
        #IP.embed()


def main():
    idmap = IDMap(sys.argv[1])


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    idmap = IDMap()




