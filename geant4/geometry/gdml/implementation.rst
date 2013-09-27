GDML Export Implementation
==========================

Intend to borrow from here for Collada export, so need to know details of
the exporter implementation.

* http://www-geant4.kek.jp/Reference/9.6.p02/classG4GDMLWrite.html

Inverted inheritance chain structure:: 

    G4GDMLWrite < ... < G4GDMLWriteStructure 

Contary to expectations from the name `G4GDMLWriteStructure` is top dog inheriting
from all the other `G4GDMLWrite*` classes including `G4GDMLWrite` at the base.
Inheritance misused for categorisation.

$DYB/NuWa-trunk/lhcb/Sim/GaussTools/src/Components/GiGaRunActionGDML.cpp::

     55    G4VPhysicalVolume* wpv = G4TransportationManager::GetTransportationManager()->
     56       GetNavigatorForTracking()->GetWorldVolume();
     57 
     58    G4String outFilePath("g4_00.gdml");
     59    G4GDMLParser parser ;
     60    if(wpv)
     61    {
     62        std::cout << "GiGaRunActionGDML::BeginOfRunAction writing to " << m_outFilePath << std::endl ;
     63        parser.Write(outFilePath, wpv);
     64    }

$DYB/external/build/LCG/geant4.9.2.p01/source/persistency/gdml/src/G4GDMLParser.cc::

     37 G4GDMLParser::G4GDMLParser()
     38   : ucode(false)
     39 {
     40   reader = new G4GDMLReadStructure;
     41   writer = new G4GDMLWriteStructure;
     42   xercesc::XMLPlatformUtils::Initialize();
     43 }

$DYB/external/build/LCG/geant4.9.2.p01/source/persistency/gdml/include/G4GDMLParser.icc::

     47 inline
     48 void G4GDMLParser::Write(const G4String& filename,
     49                          const G4VPhysicalVolume* const pvol,
     50                                G4bool refs,
     51                          const G4String& schemaLocation)
     52 {
     53   const G4int depth = 0;
     54   G4LogicalVolume* lvol = 0;
     55 
     56   if (!pvol)
     57   {
     58     G4VPhysicalVolume* worldPV = GetWorldVolume();
     59     if (!worldPV)
     60     {
     61       G4Exception("G4DMLParser::Write()", "InvalidSetup", FatalException,
     62                   "Detector-Construction needs to be registered first!");
     63     }
     64     lvol = worldPV->GetLogicalVolume();
     65   }
     66   else
     67   {
     68     lvol = pvol->GetLogicalVolume();
     69   }
     70   writer->Write(filename,lvol,schemaLocation,depth,refs);
     71 }

$DYB/external/build/LCG/geant4.9.2.p01/source/persistency/gdml/src/G4GDMLWrite.cc::

    107 G4Transform3D G4GDMLWrite::Write(const G4String& fname,
    108                                  const G4LogicalVolume* const logvol,
    109                                  const G4String& setSchemaLocation,
    110                                  const G4int depth,
    111                                        G4bool refs)
    112 {
    ///   
    ///    xercesc XML writer/doc setup
    ///
    161    DefineWrite(gdml);         // open "define" element
    162    MaterialsWrite(gdml);      // open "materials" element and clear materialsList, isotopeList 
    163    SolidsWrite(gdml);         // open "solids" element and clear solidsList
    164    StructureWrite(gdml);      // open "structure" element     
    165    SetupWrite(gdml,logvol);   // open "setup" element and populate
    166 
    167    G4Transform3D R = TraverseVolumeTree(logvol,depth);   // the meat, kicking off the recursive traverse 
    168 
    ///
    ///    xercesc XML writing 
    ///
    216    return R;
    217 }


$DYB/external/build/LCG/geant4.9.2.p01/source/persistency/gdml/src/G4GDMLWriteStructure.cc::

    189 G4Transform3D G4GDMLWriteStructure::
    190 TraverseVolumeTree(const G4LogicalVolume* const volumePtr, const G4int depth)
    191 {
    192    if (VolumeMap().find(volumePtr) != VolumeMap().end())
    193    {
    194      return VolumeMap()[volumePtr]; // Volume is already processed
    195    }
    196 
    197    G4VSolid* solidPtr = volumePtr->GetSolid();
    198    G4Transform3D R,invR;
    ///
    ///    reflected solid handling skipped
    ///
    235    const G4String name
    236          = GenerateName(volumePtr->GetName(),volumePtr);      // lvName
    237    const G4String materialref
    238          = GenerateName(volumePtr->GetMaterial()->GetName(),  
    239                         volumePtr->GetMaterial());
    240    const G4String solidref
    241          = GenerateName(solidPtr->GetName(),solidPtr);
    ...    
    ...    prep the volumeElement using name/materialref/solidref but dont append to structure yet
    ...
    ...    2181     <volume name="/dd/Geometry/RPC/lvRPCGasgap140xb7491f8">
    ...    2182       <materialref ref="/dd/Materials/Air0xb830740"/>
    ...    2183       <solidref ref="RPCGasgap140xbad5938"/>
    ... 
    ...
    232    if (reflected>0) { invR = R.inverse(); }
    233      // Only compute the inverse when necessary!
    ...
    252    const G4int daughterCount = volumePtr->GetNoDaughters();
    253 
    254    for (G4int i=0;i<daughterCount;i++)   // Traverse all the children!
    255    {
    256       const G4VPhysicalVolume* const physvol = volumePtr->GetDaughter(i);
    257       const G4String ModuleName = Modularize(physvol,depth);
    258 
    259       G4Transform3D daughterR;
    260 
    261       if (ModuleName.empty())   // Check if subtree requested to be 
    262       {                         // a separate module!
    263          daughterR = TraverseVolumeTree(physvol->GetLogicalVolume(),depth+1);
    ...
    ...   Q: hmm, how come not getting a deeper GDML structure then ?
    ...   A: because no matter what depth of the recursion the resulting volumeElement with 0 or more child physvol
    ...      are appended to the fixed structureElement gdml/structure/
    ...
    264       }
    265       else
    266       {
    267          G4GDMLWriteStructure writer;
    268          daughterR = writer.Write(ModuleName,physvol->GetLogicalVolume(),
    269                                   SchemaLocation,depth+1);
    270       }
    ...
    272       if (const G4PVDivision* const divisionvol
    273          = dynamic_cast<const G4PVDivision*>(physvol)) // Is it division?
    274       {
    ///
    ///              divisional/replica/parameterized skipped
    ///
    309       else   // Is it a physvol?
    310       {
    311          G4RotationMatrix rot;
    312 
    313          if (physvol->GetFrameRotation() != 0)
    314          {
    315            rot = *(physvol->GetFrameRotation());
    316          }
    317          G4Transform3D P(rot,physvol->GetObjectTranslation());           // placement transform of daughter pv wrt mother
    318          PhysvolWrite(volumeElement,physvol,invR*P*daughterR,ModuleName);
    ...
    ...             R, invR are identity transforms when not dealing with reflections.. ?
    ...
    319       }
    320    }
    321 
    322    structureElement->appendChild(volumeElement);
    323      // Append the volume AFTER traversing the children so that
    324      // the order of volumes will be correct!
    325 
    326    VolumeMap()[volumePtr] = R;
    327 
    328    G4GDMLWriteMaterials::AddMaterial(volumePtr->GetMaterial());
    329      // Add the involved materials and solids!
    330 
    331    G4GDMLWriteSolids::AddSolid(solidPtr);
    332 
    333    return R;
    334 }




::

     76 void G4GDMLWriteStructure::PhysvolWrite(xercesc::DOMElement* volumeElement,
     77                                         const G4VPhysicalVolume* const physvol,
     78                                         const G4Transform3D& T,
     79                                         const G4String& ModuleName)
     80 {
     81    HepGeom::Scale3D scale;
     82    HepGeom::Rotate3D rotate;
     83    HepGeom::Translate3D translate;
     84 
     85    T.getDecomposition(scale,rotate,translate);
     86 
     87    const G4ThreeVector scl(scale(0,0),scale(1,1),scale(2,2));
     88    const G4ThreeVector rot = GetAngles(rotate.getRotation());
     89    const G4ThreeVector pos = T.getTranslation();
     90 
     91    const G4String name = GenerateName(physvol->GetName(),physvol);
     92 
     93    xercesc::DOMElement* physvolElement = NewElement("physvol");
     94    physvolElement->setAttributeNode(NewAttribute("name",name));
     95    volumeElement->appendChild(physvolElement);
     96 
     97    const G4String volumeref
     98          = GenerateName(physvol->GetLogicalVolume()->GetName(),
     99                         physvol->GetLogicalVolume());
     100 
     101    if (ModuleName.empty())
     102    {
     103       xercesc::DOMElement* volumerefElement = NewElement("volumeref");
     104       volumerefElement->setAttributeNode(NewAttribute("ref",volumeref));
     105       physvolElement->appendChild(volumerefElement);
     106    }
     ...
     ...       physvol are the children of the mother lv. 
     ...       volumeref/@ref point to the lv of the child physvol
     ...
     ...       2181     <volume name="/dd/Geometry/RPC/lvRPCGasgap140xb7491f8">
     ...       2182       <materialref ref="/dd/Materials/Air0xb830740"/>
     ...       2183       <solidref ref="RPCGasgap140xbad5938"/>
     ...       2184       <physvol name="/dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:1#pvStrip14Unit0xbc1e930">
     ...       2185         <volumeref ref="/dd/Geometry/RPC/lvRPCStrip0xb839910"/>
     ...       2186         <position name="/dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:1#pvStrip14Unit0xbc1e930_pos" unit="mm" x="-910" y="0" z="0"/>
     ...       2187         <rotation name="/dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:1#pvStrip14Unit0xbc1e930_rot" unit="deg" x="0" y="0" z="-90"/>
     ...       2188       </physvol>
     ...
     ...
     107    else
     108    {
     109       xercesc::DOMElement* fileElement = NewElement("file");
     110       fileElement->setAttributeNode(NewAttribute("name",ModuleName));
     111       fileElement->setAttributeNode(NewAttribute("volname",volumeref));
     112       physvolElement->appendChild(fileElement);
     113    }
     114 
     115    if (std::fabs(pos.x()) > kLinearPrecision
     116     || std::fabs(pos.y()) > kLinearPrecision
     117     || std::fabs(pos.z()) > kLinearPrecision)
     118    {
     119      PositionWrite(physvolElement,name+"_pos",pos);
     120    }
     121    if (std::fabs(rot.x()) > kAngularPrecision
     122     || std::fabs(rot.y()) > kAngularPrecision
     123     || std::fabs(rot.z()) > kAngularPrecision)
     124    {
     125      RotationWrite(physvolElement,name+"_rot",rot);
     126    }
     127    if (std::fabs(scl.x()-1.0) > kRelativePrecision
     128     || std::fabs(scl.y()-1.0) > kRelativePrecision
     129     || std::fabs(scl.z()-1.0) > kRelativePrecision)
     130    {
     131      ScaleWrite(physvolElement,name+"_scl",scl);
     132    }
     133 }



$LOCAL_BASE/env/geant4/geometry/gdml/g4_01.gdml::

    ...
    ...     structure only goes to two levels ?  structure/volume/physvol/volumeref
    ...     presumably depth heirarchy being repesented via the volumeref linking up multiple such relations 
    ...
    ...     Logical volumes with @name are pointed to by physvol/volumeref/@ref 
    ...
    2172   <structure>
    2173     <volume name="/dd/Geometry/PoolDetails/lvNearTopCover0xbad46a0">
    2174       <materialref ref="/dd/Materials/PPE0xb8310e0"/>
    2175       <solidref ref="near_top_cover_box0xbad4490"/>
    2176     </volume>
    2177     <volume name="/dd/Geometry/RPC/lvRPCStrip0xb839910">
    2178       <materialref ref="/dd/Materials/MixGas0xbad5d28"/>
    2179       <solidref ref="RPCStrip0xb751cc0"/>
    2180     </volume>
    2181     <volume name="/dd/Geometry/RPC/lvRPCGasgap140xb7491f8">
    2182       <materialref ref="/dd/Materials/Air0xb830740"/>
    2183       <solidref ref="RPCGasgap140xbad5938"/>
    2184       <physvol name="/dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:1#pvStrip14Unit0xbc1e930">
    2185         <volumeref ref="/dd/Geometry/RPC/lvRPCStrip0xb839910"/>
    2186         <position name="/dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:1#pvStrip14Unit0xbc1e930_pos" unit="mm" x="-910" y="0" z="0"/>
    2187         <rotation name="/dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:1#pvStrip14Unit0xbc1e930_rot" unit="deg" x="0" y="0" z="-90"/>
    2188       </physvol>
    2189       <physvol name="/dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:2#pvStrip14Unit0xbc1f8b8">
    2190         <volumeref ref="/dd/Geometry/RPC/lvRPCStrip0xb839910"/>
    2191         <position name="/dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:2#pvStrip14Unit0xbc1f8b8_pos" unit="mm" x="-650" y="0" z="0"/>
    2192         <rotation name="/dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:2#pvStrip14Unit0xbc1f8b8_rot" unit="deg" x="0" y="0" z="-90"/>
    2193       </physvol>
    .... 
    2219       <physvol name="/dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:8#pvStrip14Unit0xbc1feb0">
    2220         <volumeref ref="/dd/Geometry/RPC/lvRPCStrip0xb839910"/>
    2221         <position name="/dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:8#pvStrip14Unit0xbc1feb0_pos" unit="mm" x="910" y="0" z="0"/>
    2222         <rotation name="/dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:8#pvStrip14Unit0xbc1feb0_rot" unit="deg" x="0" y="0" z="-90"/>
    2223       </physvol>
    2224     </volume>
    2225     <volume name="/dd/Geometry/RPC/lvRPCBarCham140xbad5978">
    2226       <materialref ref="/dd/Materials/Bakelite0xb830008"/>
    2227       <solidref ref="RPCBarCham140xbd59170"/>
    2228       <physvol name="/dd/Geometry/RPC/lvRPCBarCham14#pvRPCGasgap140xbc1f360">
    2229         <volumeref ref="/dd/Geometry/RPC/lvRPCGasgap140xb7491f8"/>
    2230       </physvol>
    2231     </volume>
    ....
    30919     <volume name="/dd/Geometry/Sites/lvNearSiteRock0xb82e578">
    30920       <materialref ref="/dd/Materials/Rock0xb849090"/>
    30921       <solidref ref="near_rock0xb8499c8"/>
    30922       <physvol name="/dd/Geometry/Sites/lvNearSiteRock#pvNearHallTop0xb7dd068">
    30923         <volumeref ref="/dd/Geometry/Sites/lvNearHallTop0xb745f10"/>
    30924         <position name="/dd/Geometry/Sites/lvNearSiteRock#pvNearHallTop0xb7dd068_pos" unit="mm" x="2500" y="-500" z="7500"/>
    30925       </physvol>
    30926       <physvol name="/dd/Geometry/Sites/lvNearSiteRock#pvNearHallBot0xc5065d0">
    30927         <volumeref ref="/dd/Geometry/Sites/lvNearHallBot0xb7dd4a8"/>
    30928         <position name="/dd/Geometry/Sites/lvNearSiteRock#pvNearHallBot0xc5065d0_pos" unit="mm" x="0" y="0" z="-5150"/>
    30929       </physvol>
    30930     </volume>
    30931     <volume name="World0xc6337a8">
    30932       <materialref ref="/dd/Materials/Vacuum0xbaff828"/>
    30933       <solidref ref="WorldBox0xc6328f0"/>
    30934       <physvol name="/dd/Structure/Sites/db-rock0xc633af8">
    30935         <volumeref ref="/dd/Geometry/Sites/lvNearSiteRock0xb82e578"/>
    30936         <position name="/dd/Structure/Sites/db-rock0xc633af8_pos" unit="mm" x="-16519.9999999999" y="-802110" z="-2110"/>
    30937         <rotation name="/dd/Structure/Sites/db-rock0xc633af8_rot" unit="deg" x="0" y="0" z="-122.9"/>
    30938       </physvol>
    30939     </volume>
    30940   </structure>
 


$LOCAL_BASE/env/geant4/geometry/gdml/g4_01.gdml::

    1 <?xml version="1.0" encoding="UTF-8" standalone="no" ?>
    2 <gdml xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://service-spi.web.cern.ch/service-spi/app/releases/GDML/schema/gdml.xsd">
    3 
    4   <define/>
    5 
    6   <materials>
    7     <element Z="6" name="/dd/Materials/Carbon0xbad48f8">
    8       <atom unit="g/mole" value="12.0109936803044"/>
    9     </element>
    10     <element Z="1" name="/dd/Materials/Hydrogen0xbad34d0">
    11       <atom unit="g/mole" value="1.00793946966331"/>
    12     </element>
    13     <material name="/dd/Materials/PPE0xb8310e0" state="solid">
    14       <P unit="pascal" value="101324.946686941"/>
    15       <D unit="g/cm3" value="0.919999515933733"/>
    16       <fraction n="0.798874855998063" ref="/dd/Materials/Carbon0xbad48f8"/>
    17       <fraction n="0.201125144001937" ref="/dd/Materials/Hydrogen0xbad34d0"/>
    18     </material>    
    ..
    401   <solids>
    402     <box lunit="mm" name="near_top_cover0xbb2aa88" x="16000" y="10000" z="44"/>
    403     <box lunit="mm" name="near_top_cover_sub00xbad4c78" x="4249.00272282321" y="4249.00272282321" z="54"/>
    404     <subtraction name="near_top_cover-ChildFornear_top_cover_box0xbad6708">
    405       <first ref="near_top_cover0xbb2aa88"/>
    406       <second ref="near_top_cover_sub00xbad4c78"/>
    407       <position name="near_top_cover-ChildFornear_top_cover_box0xbad6708_pos" unit="mm" x="8000" y="5000" z="0"/>
    408       <rotation name="near_top_cover-ChildFornear_top_cover_box0xbad6708_rot" unit="deg" x="0" y="0" z="45"/>
    409     </subtraction>
    ...
    2172   <structure>
    ....
    ....     dealt with above
    ....
    30940   </structure>
    30941 
    30942   <setup name="Default" version="1.0">
    30943     <world ref="World0xc6337a8"/>
    30944   </setup>
    30945 
    30946 </gdml>

          
