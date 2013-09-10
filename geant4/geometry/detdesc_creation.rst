DetDesc Creation of Geometry
=============================

::

    DetectorPersistencySvc                INFO  'CnvServices':[ 'XmlCnvSvc/XmlCnvSvc' ]
    DetectorPersistencySvc                INFO Added successfully Conversion service:XmlCnvSvc
    DetectorDataSvc                    SUCCESS Detector description database: /data1/env/local/dyb/NuWa-trunk/dybgaudi/Detector/XmlDetDesc/DDDB/dayabay.xml
    EventClockSvc.FakeEventTime           INFO Event times generated from 0 with steps of 0



DetDesc Geometry Dumping
-------------------------- 


* Run XmlDetDescCheck's XddDumpAlg

   * :dybsvn:`source:dybgaudi/trunk/Detector/XmlDetDesc/python/XmlDetDesc/dump_geo.py`



::

    [blyth@belle7 ~]$ nuwa.py -G $XMLDETDESCROOT/DDDB/dayabay.xml -n1 -m XmlDetDesc.dump_geo > dayabay_dump.out
    Warning in <TEnvRec::ChangeValue>: duplicate entry <Library.vector<char>=vector.dll> for level 0; ignored
    Warning in <TEnvRec::ChangeValue>: duplicate entry <Library.vector<short>=vector.dll> for level 0; ignored
    Warning in <TEnvRec::ChangeValue>: duplicate entry <Library.vector<unsigned-int>=vector.dll> for level 0; ignored
    [blyth@belle7 ~]$ du -h dayabay_dump.out
    688K    dayabay_dump.out


::

    [blyth@belle7 ~]$ fenv
    [blyth@belle7 ~]$ nuwa.py -G $XMLDETDESCROOT/DDDB/radslabs.xml -n1 -m XmlDetDesc.dump_geo
    ...
    EventLoopMgr                          INFO No event selector to be used
    ApplicationMgr                        INFO Application Manager Initialized successfully
    ApplicationMgr                        INFO Application Manager Started successfully
    Dumping /dd/Geometry
    DetectorPersistencySvc                INFO  'CnvServices':[ 'XmlCnvSvc/XmlCnvSvc' ]
    DetectorPersistencySvc                INFO Added successfully Conversion service:XmlCnvSvc
    DetectorDataSvc                    SUCCESS Detector description database: /data1/env/local/dyb/NuWa-trunk/dybgaudi/Detector/XmlDetDesc/DDDB/radslabs.xml
    /Geometry: DataObject at 0xb8f0600
    1 sub-directories:
     Registry: name=/RadSlabs, identifier=/dd/Geometry/RadSlabs
      /RadSlabs: DataObject at 0xb8fb6e8
      14 sub-directories:
       Registry: name=/lvNearRadSlab1, identifier=/dd/Geometry/RadSlabs/lvNearRadSlab1
        /lvNearRadSlab1: DataObject at 0xb913c28
    Validity: 0.0 -> 9223372036.854775807
         LVolume (1)  name = '/dd/Geometry/RadSlabs/lvNearRadSlab1'  #physvols0
     SolidType='SolidBox'   name='near-radslab-box-1'
     BPs: (x,y,z,r,rho)[Min/Max][mm]=(     -150/      150,   -2e+03/    2e+03,   -5e+03/    5e+03, 5.39e+03,    2e+03) 
    [ xsize[mm]=      300 ysize[mm]= 3.99e+03 zsize[mm]=    1e+04]

    Material name='/dd/Materials/RadRock' 

        0 sub-directories:
       Registry: name=/lvNearRadSlab2, identifier=/dd/Geometry/RadSlabs/lvNearRadSlab2
        /lvNearRadSlab2: DataObject at 0xb913e30
    Validity: 0.0 -> 9223372036.854775807
         LVolume (2)  name = '/dd/Geometry/RadSlabs/lvNearRadSlab2'  #physvols0
     SolidType='SolidBox'   name='near-radslab-box-2'
     BPs: (x,y,z,r,rho)[Min/Max][mm]=(     -150/      150,-2.12e+03/ 2.12e+03,   -5e+03/    5e+03, 5.43e+03, 2.13e+03) 
    [ xsize[mm]=      300 ysize[mm]= 4.25e+03 zsize[mm]=    1e+04]

    Material name='/dd/Materials/RadRock' 

        0 sub-directories:
       Registry: name=/lvNearRadSlab3, identifier=/dd/Geometry/RadSlabs/lvNearRadSlab3
        /lvNearRadSlab3: DataObject at 0xb913f90
    Validity: 0.0 -> 9223372036.854775807
         LVolume (3)  name = '/dd/Geometry/RadSlabs/lvNearRadSlab3'  #physvols0
     SolidType='SolidBox'   name='near-radslab-box-3'
     BPs: (x,y,z,r,rho)[Min/Max][mm]=(     -150/      150,   -5e+03/    5e+03,   -5e+03/    5e+03, 7.07e+03,    5e+03) 
    [ xsize[mm]=      300 ysize[mm]= 9.99e+03 zsize[mm]=    1e+04]


    ...

    Material name='/dd/Materials/RadRock' 

        0 sub-directories:
       Registry: name=/lvNearRadSlab9, identifier=/dd/Geometry/RadSlabs/lvNearRadSlab9
        /lvNearRadSlab9: DataObject at 0xb9147b0
    Validity: 0.0 -> 9223372036.854775807
         LVolume (9)  name = '/dd/Geometry/RadSlabs/lvNearRadSlab9'  #physvols0
     SolidType='SolidSubtraction'   name='near-radslab-box-9'
     BPs: (x,y,z,r,rho)[Min/Max][mm]=(   -8e+03/    8e+03,   -5e+03/    5e+03,     -150/      150, 9.44e+03, 9.43e+03) 
     ** 'Main' solid is 
     SolidType='SolidBox'   name='near-radslab-box-9-box'
     BPs: (x,y,z,r,rho)[Min/Max][mm]=(   -8e+03/    8e+03,   -5e+03/    5e+03,     -150/      150, 9.44e+03, 9.43e+03) 
    [ xsize[mm]=  1.6e+04 ysize[mm]=    1e+04 zsize[mm]=      300]
     ** 'Booled' with 
     SolidType='SolidChild'         name='Child For near-radslab-box-9'
     BPs: (x,y,z,r,rho)[Min/Max][mm]=( 4.99e+03/  1.1e+04, 1.99e+03/ 8.01e+03,-3.01e+03/ 3.01e+03, 1.24e+04, 1.24e+04) 
     SolidType='SolidBox'   name='near-radslab-box-9-sub0'
     BPs: (x,y,z,r,rho)[Min/Max][mm]=(-2.12e+03/ 2.12e+03,-2.12e+03/ 2.12e+03,     -155/      155, 3.01e+03,    3e+03) 
    [ xsize[mm]= 4.25e+03 ysize[mm]= 4.25e+03 zsize[mm]=      310]
     ** 'Booled' with 
     SolidType='SolidChild'         name='Child For near-radslab-box-9'
     BPs: (x,y,z,r,rho)[Min/Max][mm]=( 4.99e+03/  1.1e+04,-8.01e+03/-1.99e+03,-3.01e+03/ 3.01e+03, 1.24e+04, 1.24e+04) 
     SolidType='SolidBox'   name='near-radslab-box-9-sub1'
     BPs: (x,y,z,r,rho)[Min/Max][mm]=(-2.12e+03/ 2.12e+03,-2.12e+03/ 2.12e+03,     -155/      155, 3.01e+03,    3e+03) 
    [ xsize[mm]= 4.25e+03 ysize[mm]= 4.25e+03 zsize[mm]=      310]
     ** 'Booled' with 
     SolidType='SolidChild'         name='Child For near-radslab-box-9'
     BPs: (x,y,z,r,rho)[Min/Max][mm]=( -1.1e+04/-4.99e+03, 1.99e+03/ 8.01e+03,-3.01e+03/ 3.01e+03, 1.24e+04, 1.24e+04) 
     SolidType='SolidBox'   name='near-radslab-box-9-sub2'
     BPs: (x,y,z,r,rho)[Min/Max][mm]=(-2.12e+03/ 2.12e+03,-2.12e+03/ 2.12e+03,     -155/      155, 3.01e+03,    3e+03) 
    [ xsize[mm]= 4.25e+03 ysize[mm]= 4.25e+03 zsize[mm]=      310]
     ** 'Booled' with 
     SolidType='SolidChild'         name='Child For near-radslab-box-9'
     BPs: (x,y,z,r,rho)[Min/Max][mm]=( -1.1e+04/-4.99e+03,-8.01e+03/-1.99e+03,-3.01e+03/ 3.01e+03, 1.24e+04, 1.24e+04) 
     SolidType='SolidBox'   name='near-radslab-box-9-sub3'
     BPs: (x,y,z,r,rho)[Min/Max][mm]=(-2.12e+03/ 2.12e+03,-2.12e+03/ 2.12e+03,     -155/      155, 3.01e+03,    3e+03) 
    [ xsize[mm]= 4.25e+03 ysize[mm]= 4.25e+03 zsize[mm]=      310]


    Material name='/dd/Materials/RadRock' 

        0 sub-directories:
       Registry: name=/lvNearHallRadSlabs, identifier=/dd/Geometry/RadSlabs/lvNearHallRadSlabs
        /lvNearHallRadSlabs: DataObject at 0xb9148a0
    Validity: 0.0 -> 9223372036.854775807
         LAssembly (10)  name = '/dd/Geometry/RadSlabs/lvNearHallRadSlabs'  #physvols9#0  class PVolume (9) [ name='pvNearHallRadSlab1' logvol='/dd/Geometry/RadSlabs/lvNearRadSlab1']
    #1  class PVolume (9) [ name='pvNearHallRadSlab2' logvol='/dd/Geometry/RadSlabs/lvNearRadSlab2']
    #2  class PVolume (9) [ name='pvNearHallRadSlab3' logvol='/dd/Geometry/RadSlabs/lvNearRadSlab3']
    #3  class PVolume (9) [ name='pvNearHallRadSlab4' logvol='/dd/Geometry/RadSlabs/lvNearRadSlab4']
    #4  class PVolume (9) [ name='pvNearHallRadSlab5' logvol='/dd/Geometry/RadSlabs/lvNearRadSlab5']
    #5  class PVolume (9) [ name='pvNearHallRadSlab6' logvol='/dd/Geometry/RadSlabs/lvNearRadSlab6']
    #6  class PVolume (9) [ name='pvNearHallRadSlab7' logvol='/dd/Geometry/RadSlabs/lvNearRadSlab7']
    #7  class PVolume (9) [ name='pvNearHallRadSlab8' logvol='/dd/Geometry/RadSlabs/lvNearRadSlab8']
    #8  class PVolume (9) [ name='pvNearHallRadSlab9' logvol='/dd/Geometry/RadSlabs/lvNearRadSlab9']


    Material name='/dd/Materials/RadRock' 

        0 sub-directories:
       Registry: name=/lvFarHallRadSlabs, identifier=/dd/Geometry/RadSlabs/lvFarHallRadSlabs
        /lvFarHallRadSlabs: DataObject at 0xb8e9350
    Validity: 0.0 -> 9223372036.854775807
         LAssembly (20)  name = '/dd/Geometry/RadSlabs/lvFarHallRadSlabs'  #physvols9#0  class PVolume (18) [ name='pvFarHallRadSlab1' logvol='/dd/Geometry/RadSlabs/lvFarRadSlab1']
    #1  class PVolume (18) [ name='pvFarHallRadSlab2' logvol='/dd/Geometry/RadSlabs/lvFarRadSlab2']
    #2  class PVolume (18) [ name='pvFarHallRadSlab3' logvol='/dd/Geometry/RadSlabs/lvFarRadSlab3']
    #3  class PVolume (18) [ name='pvFarHallRadSlab4' logvol='/dd/Geometry/RadSlabs/lvFarRadSlab4']
    #4  class PVolume (18) [ name='pvFarHallRadSlab5' logvol='/dd/Geometry/RadSlabs/lvFarRadSlab5']
    #5  class PVolume (18) [ name='pvFarHallRadSlab6' logvol='/dd/Geometry/RadSlabs/lvFarRadSlab6']
    #6  class PVolume (18) [ name='pvFarHallRadSlab7' logvol='/dd/Geometry/RadSlabs/lvFarRadSlab7']
    #7  class PVolume (18) [ name='pvFarHallRadSlab8' logvol='/dd/Geometry/RadSlabs/lvFarRadSlab8']
    #8  class PVolume (18) [ name='pvFarHallRadSlab9' logvol='/dd/Geometry/RadSlabs/lvFarRadSlab9']


        0 sub-directories:
    ApplicationMgr                        INFO Application Manager Stopped successfully
    ApplicationMgr                        INFO Application Manager Finalized successfully
    ApplicationMgr                        INFO Application Manager Terminated successfully

