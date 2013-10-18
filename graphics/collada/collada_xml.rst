Collada XML layout
====================

::

    import os
    import lxml.etree as ET
    parse_ = lambda _:ET.parse(os.path.expandvars(_)).getroot()
    COLLADA_NS = "http://www.collada.org/2005/11/COLLADASchema"
    xml = parse_("$LOCAL_BASE/env/geant4/geometry/xdae/g4_01.dae")

::

    In [249]: len(xml.findall(".//*"))
    Out[249]: 24776


Element examination
---------------------

library_nodes
~~~~~~~~~~~~~

::

    libnodes = xml.findall("{%s}library_nodes" % COLLADA_NS )[0]
    libnodeurl = [node.attrib['id'] for node in libnodes.findall("*")]

    In [292]: len(libnodeurl)
    Out[292]: 249

    In [293]: len(set(libnodeurl))
    Out[293]: 249

instance_node 
~~~~~~~~~~~~~~
    
::

    urls = [nn.attrib['url'][1:] for nn in xml.findall(".//{%s}instance_node" % COLLADA_NS )]

    In [253]: len(urls)
    Out[253]: 5643

    In [281]: len(set(urls))   
    Out[281]: 249

    In [283]: set(urls)
    Out[283]: 
    set(['__dd__Geometry__AdDetails__lvGasDistributionBoxE0xa904a88',
         '__dd__Geometry__CalibrationSources__lvGe68AirTop0xa8fee28',
         '__dd__Geometry__PoolDetails__lvTopCornerCableTray0xa9880d0',
         '__dd__Geometry__PoolDetails__lvInnShortParCableTray0xa9075d0',
         '__dd__Geometry__AdDetails__lvGDBTopFlange0xa9048b0',
         '__dd__Geometry__CalibrationBox__lvGdLSInCalibTubAbvLid0xa9028d8',
    ...

    In [294]: set(libnodeurl) == set(urls)   # all the 249 urls are used from instance_node referents
    Out[294]: True


Only 249 distinct url references, corresponding to logical volumes.
So, for subcopying need to add instance_node referents to library_nodes.


node
~~~~~~

::

    nodeid = [n.attrib['id'] for n in xml.findall(".//{%s}node" % COLLADA_NS )]

    In [274]: len(nodeid)
    Out[274]: 5892

    In [282]: len(set(nodeid))
    Out[282]: 5892

    In [284]: set(nodeid)
    Out[284]: 
    set(['__dd__Geometry__RPCSupport__lvNearHbeamBigUnit--pvNearThwartLongAIRightDownY60xa8cc388',
         '__dd__Geometry__Pool__lvNearPoolOWS--pvVetoPmtNearOutFacein--pvNearOutFaceinWall5--pvNearOutFaceinWall5..8--pvVetoPmtUnit--pvPmtMount--pvMountRib1s--pvMountRib1s..2--pvMountRib1unit0xa9c29e8',
         '__dd__Geometry__RPCSupport__lvNearHbeamSmallUnit--pvNearThwartLongAIUpY20xa8c6928',
         '__dd__Geometry__Pool__lvNearPoolOWS--pvVetoPmtNearOutFacein--pvNearOutFaceinWall9--pvNearOutFaceinWall9..4--pvVetoPmtUnit--pvPmtMount--pvMountRib3s--pvMountRib3s..2--pvMountRib3unit0xa9ed818',
         '__dd__Geometry__Pool__lvNearPoolOWS--pvVetoPmtNearOutFacein--pvNearOutFaceinWall3--pvNearOutFaceinWall3..1--pvVetoPmtUnit--pvPmtMount--pvMountRib1s--pvMountRib1s..1--pvMountRib1unit0xa99b208',
         '__dd__Geometry__Pool__lvNearPoolOWS--pvVetoPmtNearOutFacein--pvNearOutFaceinWall8--pvNearOutFaceinWall8..7--pvVetoPmtUnit--pvPmtMount--pvMountRib3s--pvMountRib3s..1--pvMountRib3unit0xa9dcc58',
         '__dd__Geometry__Pool__lvNearPoolIWS--pvVetoPmtNearInn--pvNearInnWall7--pvNearInnWall7..10--pvVetoPmtUnit--pvPmtMount--pvMountRib3s--pvMountRib3s..1--pvMountRib3unit0xa958630',
         ...

    In [302]: set([len(n.findall("*")) for  n in xml.findall(".//{%s}node" % COLLADA_NS )])
    Out[302]: set([1, 2, 3, 36, 5, 6, 7, 73, 10, 11, 12, 1620, 179, 9, 521, 55, 4, 2939])    # distinct child counts for every node

    In [308]: print len([ET.tostring(n) for n in filter(lambda _:len(_.findall("*"))==1, xml.findall(".//{%s}node" % COLLADA_NS ))])
    165

    In [307]: print "\n".join([ET.tostring(n) for n in filter(lambda _:len(_.findall("*"))==1, xml.findall(".//{%s}node" % COLLADA_NS ))])     # all 1 child nodes, mostly instance_geometry 

    <node xmlns="http://www.collada.org/2005/11/COLLADASchema" id="__dd__Geometry__PoolDetails__lvNearTopCover0xa8bffe8">
          <instance_geometry url="#near_top_cover_box0x8869f48">
            <bind_material>
              <technique_common>
                <instance_material symbol="WHITE" target="#__dd__Materials__PPE0x88166b8"/>
              </technique_common>
            </bind_material>
          </instance_geometry>
        </node>
        
    <node xmlns="http://www.collada.org/2005/11/COLLADASchema" id="__dd__Geometry__RPC__lvRPCStrip0xa8c01d8">
          <instance_geometry url="#RPCStrip0x886a088">
            <bind_material>
              <technique_common>
                <instance_material symbol="WHITE" target="#__dd__Materials__MixGas0x8837740"/>
              </technique_common>
            </bind_material>
          </instance_geometry>
        </node>

    ...
    
    <node xmlns="http://www.collada.org/2005/11/COLLADASchema" id="top">
            <instance_node url="#World0xaa8afb8"/>
          </node>
        

    In [311]: print "\n".join([ET.tostring(n) for n in filter(lambda _:len(_.findall("*"))==2, xml.findall(".//{%s}node" % COLLADA_NS ))][0:1])
    <node xmlns="http://www.collada.org/2005/11/COLLADASchema" id="__dd__Geometry__RPC__lvRPCGasgap14--pvStrip14Array--pvStrip14ArrayOne..1--pvStrip14Unit0xa8c02c0">
            <matrix>
                                    6.12303e-17 1 0 -910
    -1 6.12303e-17 0 0
    0 0 1 0
    0.0 0.0 0.0 1.0
    </matrix>
            <instance_node url="#__dd__Geometry__RPC__lvRPCStrip0xa8c01d8"/>
          </node>
          

    In [313]: print "\n".join([ET.tostring(n) for n in filter(lambda _:len(_.findall("*"))==2, xml.findall(".//{%s}node" % COLLADA_NS ))][100:101])
    <node xmlns="http://www.collada.org/2005/11/COLLADASchema" id="__dd__Geometry__RPCSupport__lvNearHbeamSmallUnit--pvNearThwartLongAIDownY80xa8c7618">
            <matrix>
                                    1 0 0 0
    0 1 0 3695
    0 0 1 -143
    0.0 0.0 0.0 1.0
    </matrix>
            <instance_node url="#__dd__Geometry__RPCSupport__TrivialComponents__lvNearThwartLongAngleIron0xa8c6128"/>
          </node>
          


::

    In [325]: print len([ET.tostring(n) for n in filter(lambda _:len(_.findall("*"))==1, xml.findall(".//{%s}node" % COLLADA_NS ))])
    165

    In [314]: print len([ET.tostring(n) for n in filter(lambda _:len(_.findall("*"))==2, xml.findall(".//{%s}node" % COLLADA_NS ))])
    5683

    In [315]: print len([ET.tostring(n) for n in filter(lambda _:len(_.findall("*"))==3, xml.findall(".//{%s}node" % COLLADA_NS ))])
    9

    In [316]: print len([ET.tostring(n) for n in filter(lambda _:len(_.findall("*"))==4, xml.findall(".//{%s}node" % COLLADA_NS ))])
    4

    In [317]: print len([ET.tostring(n) for n in filter(lambda _:len(_.findall("*"))==5, xml.findall(".//{%s}node" % COLLADA_NS ))])
    5

    In [318]: print len([ET.tostring(n) for n in filter(lambda _:len(_.findall("*"))==6, xml.findall(".//{%s}node" % COLLADA_NS ))])
    6

    In [319]: print len([ET.tostring(n) for n in filter(lambda _:len(_.findall("*"))==7, xml.findall(".//{%s}node" % COLLADA_NS ))])
    5

    In [320]: print len([ET.tostring(n) for n in filter(lambda _:len(_.findall("*"))==8, xml.findall(".//{%s}node" % COLLADA_NS ))])
    0

    In [321]: print len([ET.tostring(n) for n in filter(lambda _:len(_.findall("*"))==9, xml.findall(".//{%s}node" % COLLADA_NS ))])
    2

    In [322]: print len([ET.tostring(n) for n in filter(lambda _:len(_.findall("*"))==10, xml.findall(".//{%s}node" % COLLADA_NS ))])
    3

    In [323]: print len([ET.tostring(n) for n in filter(lambda _:len(_.findall("*"))==11, xml.findall(".//{%s}node" % COLLADA_NS ))])
    1

    In [324]: print len([ET.tostring(n) for n in filter(lambda _:len(_.findall("*"))==12, xml.findall(".//{%s}node" % COLLADA_NS ))])
    2




All nodeid are distinct, corresponding to physical volumes.



Eleven child example
~~~~~~~~~~~~~~~~~~~~~~

::

    In [326]: print "\n".join([ET.tostring(n) for n in filter(lambda _:len(_.findall("*"))==11, xml.findall(".//{%s}node" % COLLADA_NS ))])
    <node xmlns="http://www.collada.org/2005/11/COLLADASchema" id="__dd__Geometry__Sites__lvNearHallBot0xaa8a588">
          <instance_geometry url="#near_hall_bot0xa8bfad8">
            <bind_material>
              <technique_common>
                <instance_material symbol="WHITE" target="#__dd__Materials__Rock0x8868188"/>
              </technique_common>
            </bind_material>
          </instance_geometry>
          <node id="__dd__Geometry__Sites__lvNearHallBot--pvNearPoolDead0xaa8a0b0">
            <matrix>
                                    1 0 0 0
    0 1 0 0
    0 0 1 150
    0.0 0.0 0.0 1.0
    </matrix>
            <instance_node url="#__dd__Geometry__Pool__lvNearPoolDead0xaa898b0"/>
          </node>
          <node id="__dd__Geometry__Sites__lvNearHallBot--pvNearHallRadSlabs--pvNearHallRadSlab10xaa8a678">
            <matrix>
                                    1 0 0 8150
    0 1 0 0
    0 0 1 0
    0.0 0.0 0.0 1.0
    </matrix>
            <instance_node url="#__dd__Geometry__RadSlabs__lvNearRadSlab10xaa8a270"/>
          </node>
          <node id="__dd__Geometry__Sites__lvNearHallBot--pvNearHallRadSlabs--pvNearHallRadSlab20xaa8a730">
            <matrix>
                                    0.707107 0.707107 0 6603.82
    -0.707107 0.707107 0 3603.82
    0 0 1 0
    0.0 0.0 0.0 1.0
    </matrix>
            <instance_node url="#__dd__Geometry__RadSlabs__lvNearRadSlab20xaa8a2c8"/>
          </node>
          <node id="__dd__Geometry__Sites__lvNearHallBot--pvNearHallRadSlabs--pvNearHallRadSlab30xaa8a7e8">
            <matrix>
                                    6.12303e-17 1 0 0
    -1 6.12303e-17 0 5150
    0 0 1 0
    0.0 0.0 0.0 1.0
    </matrix>
            <instance_node url="#__dd__Geometry__RadSlabs__lvNearRadSlab30xaa8a320"/>
          </node>
          <node id="__dd__Geometry__Sites__lvNearHallBot--pvNearHallRadSlabs--pvNearHallRadSlab40xaa8a820">
            <matrix>
                                    -0.707107 0.707107 0 -6603.82
    -0.707107 -0.707107 0 3603.82
    0 0 1 0
    0.0 0.0 0.0 1.0
    </matrix>
            <instance_node url="#__dd__Geometry__RadSlabs__lvNearRadSlab40xaa8a378"/>
          </node>
          <node id="__dd__Geometry__Sites__lvNearHallBot--pvNearHallRadSlabs--pvNearHallRadSlab50xaa8a908">
            <matrix>
                                    -1 1.22461e-16 0 -8150
    -1.22461e-16 -1 0 0
    0 0 1 0
    0.0 0.0 0.0 1.0
    </matrix>
            <instance_node url="#__dd__Geometry__RadSlabs__lvNearRadSlab50xaa8a3d0"/>
          </node>
          <node id="__dd__Geometry__Sites__lvNearHallBot--pvNearHallRadSlabs--pvNearHallRadSlab60xaa8a940">
            <matrix>
                                    -0.707107 -0.707107 0 -6603.82
    0.707107 -0.707107 0 -3603.82
    0 0 1 0
    0.0 0.0 0.0 1.0
    </matrix>
            <instance_node url="#__dd__Geometry__RadSlabs__lvNearRadSlab60xaa8a428"/>
          </node>
          <node id="__dd__Geometry__Sites__lvNearHallBot--pvNearHallRadSlabs--pvNearHallRadSlab70xaa8aad8">
            <matrix>
                                    6.12303e-17 -1 0 0
    1 6.12303e-17 0 -5150
    0 0 1 0
    0.0 0.0 0.0 1.0
    </matrix>
            <instance_node url="#__dd__Geometry__RadSlabs__lvNearRadSlab70xaa8a480"/>
          </node>
          <node id="__dd__Geometry__Sites__lvNearHallBot--pvNearHallRadSlabs--pvNearHallRadSlab80xaa8ab10">
            <matrix>
                                    0.707107 -0.707107 0 6603.82
    0.707107 0.707107 0 -3603.82
    0 0 1 0
    0.0 0.0 0.0 1.0
    </matrix>
            <instance_node url="#__dd__Geometry__RadSlabs__lvNearRadSlab80xaa8a4d8"/>
          </node>
          <node id="__dd__Geometry__Sites__lvNearHallBot--pvNearHallRadSlabs--pvNearHallRadSlab90xaa8aca8">
            <matrix>
                                    1 0 0 0
    0 1 0 0
    0 0 1 -5150
    0.0 0.0 0.0 1.0
    </matrix>
            <instance_node url="#__dd__Geometry__RadSlabs__lvNearRadSlab90xaa8a530"/>
          </node>
        </node>




