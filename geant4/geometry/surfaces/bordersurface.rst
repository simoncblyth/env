G4LogicalBorderSurface
=======================


GDML/G4DAE persisted form
----------------------------

::

    157610       <bordersurface name="__dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceTop" surfaceproperty="__dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceTop">
    157611         <physvolref ref="__dd__Geometry__AdDetails__lvTopReflector--pvTopRefGap0xabcc228"/>
    157612         <physvolref ref="__dd__Geometry__AdDetails__lvTopRefGap--pvTopESR0xab4bd50"/>
    157613       </bordersurface>
    157614       <bordersurface name="__dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceBot" surfaceproperty="__dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceBot">
    157615         <physvolref ref="__dd__Geometry__AdDetails__lvBotReflector--pvBotRefGap0xaa6e3d8"/>
    157616         <physvolref ref="__dd__Geometry__AdDetails__lvBotRefGap--pvBotESR0xae4eda0"/>
    157617       </bordersurface>



Issues
~~~~~~~~

#. no copy numbers on the PV ref attributes, means not unique ? 

   * http://www-zeuthen.desy.de/geant4/geant4.9.3.b01/classG4LogicalBorderSurface.html
   * http://www-zeuthen.desy.de/geant4/geant4.9.3.b01/classG4PVPlacement.html
   * maybe should be using G4PVPlacement which implements G4VPhysicalVolume in order to have a CopyNo to give a unique ID 


Refs
-----

* :google:`G4LogicalBorderSurface`

* http://hypernews.slac.stanford.edu/HyperNews/geant4/get/docsexamples/263.html

* http://hypernews.slac.stanford.edu/HyperNews/geant4/get/opticalphotons/428.html

  * suggests need to double up G4LogicalBorderSurface with volumes switched if want photons from
    either direction to see the same surface

* http://geant4.in2p3.fr/2005/Workshop/ShortCourse/session4/P.Gumplinger.pdf




    Hmm physvolref/@ref attributes are PV names, these cannot directly 
    be matched against `node.id` as that has a uniquing count tacked on. 
    Using pvlookup reveals that cannot match to precise PV in many cases
    getting two possibilites one from each of the 2 AD.  

    Maybe need to change G4DAE to pluck the uid at C++ level ? Or 
    could be bug in BorderSurface creation ? Persisting has lost 
    the association.



PV Ambiguity Issue
--------------------

::

        dump_bordersurface

        [00] <BorderSurface AdDetails__AdSurfacesAll__ESRAirSurfaceTop REFLECTIVITY >

             pv1 (2) AdDetails__lvTopReflector--pvTopRefGap0xabcc228 
               __dd__Geometry__AdDetails__lvTopReflector--pvTopRefGap0xabcc228.0             __dd__Materials__Air0xab09580 
               __dd__Geometry__AdDetails__lvTopReflector--pvTopRefGap0xabcc228.1             __dd__Materials__Air0xab09580 

             pv2 (2) AdDetails__lvTopRefGap--pvTopESR0xab4bd50 
               __dd__Geometry__AdDetails__lvTopRefGap--pvTopESR0xab4bd50.0             __dd__Materials__ESR0xaeaaeb8 
               __dd__Geometry__AdDetails__lvTopRefGap--pvTopESR0xab4bd50.1             __dd__Materials__ESR0xaeaaeb8 


            Oil http://belle7.nuu.edu.tw/dae/tree/__dd__Geometry__AD__lvSST--pvOIL0xaa6d998.0.html
                http://belle7.nuu.edu.tw/dae/tree/3155.html  (many children)

            Acr http://belle7.nuu.edu.tw/dae/tree/__dd__Geometry__AD__lvOIL--pvTopReflector0xab22490.0.html
                http://belle7.nuu.edu.tw/dae/tree/4425.html    (Acrylic, single child)

            pv1 http://belle7.nuu.edu.tw/dae/tree/__dd__Geometry__AdDetails__lvTopReflector--pvTopRefGap0xabcc228.0___4.html
            pv1 http://belle7.nuu.edu.tw/dae/tree/4426___4.html  (Air, single child)

            pv2 http://belle7.nuu.edu.tw/dae/tree/__dd__Geometry__AdDetails__lvTopRefGap--pvTopESR0xab4bd50.0.html
            pv2 http://belle7.nuu.edu.tw/dae/tree/4427.html   (EST, leaf )
          

            http://belle7.nuu.edu.tw/dae/tree/__dd__Geometry__AdDetails__lvTopReflector--pvTopRefGap0xabcc228.1___4.html
            http://belle7.nuu.edu.tw/dae/tree/6086___4.html
            http://belle7.nuu.edu.tw/dae/tree/__dd__Geometry__AdDetails__lvTopRefGap--pvTopESR0xab4bd50.1.html
            http://belle7.nuu.edu.tw/dae/tree/6087.html
            
            This bordersurface pair are (single-parent)-(single-child) with the child being leaf node
            The PV ambiguity is between the two ADs.
            Construction is simarly shaped discs 
            
                      Oil-Acrylic-Air-ESR
                                  pv1 pv2

            Double ambiguity, should yield two border surfaces ... the parent child pairings
            can be used to break ambiguity ?


        [01] <BorderSurface AdDetails__AdSurfacesAll__ESRAirSurfaceBot REFLECTIVITY >
             pv1 (2) AdDetails__lvBotReflector--pvBotRefGap0xaa6e3d8 
               __dd__Geometry__AdDetails__lvBotReflector--pvBotRefGap0xaa6e3d8.0             __dd__Materials__Air0xab09580 
               __dd__Geometry__AdDetails__lvBotReflector--pvBotRefGap0xaa6e3d8.1             __dd__Materials__Air0xab09580 
             pv2 (2) AdDetails__lvBotRefGap--pvBotESR0xae4eda0 
               __dd__Geometry__AdDetails__lvBotRefGap--pvBotESR0xae4eda0.0             __dd__Materials__ESR0xaeaaeb8 
               __dd__Geometry__AdDetails__lvBotRefGap--pvBotESR0xae4eda0.1             __dd__Materials__ESR0xaeaaeb8 

             Presumably same pattern as top reflector 

             Double ambiguity, means this should yield two surfaces... one for each AD


        [02] <BorderSurface AdDetails__AdSurfacesAll__SSTOilSurface REFLECTIVITY >
             pv1 (2) AD__lvSST--pvOIL0xaa6d998 
               __dd__Geometry__AD__lvSST--pvOIL0xaa6d998.0             __dd__Materials__MineralOil0xaecfd78 
               __dd__Geometry__AD__lvSST--pvOIL0xaa6d998.1             __dd__Materials__MineralOil0xaecfd78 

               http://belle7.nuu.edu.tw/dae/tree/__dd__Geometry__AD__lvSST--pvOIL0xaa6d998.0.html
               http://belle7.nuu.edu.tw/dae/tree/3155.html   Oil
               
             pv2 (2) AD__lvADE--pvSST0xaba3f60 
               __dd__Geometry__AD__lvADE--pvSST0xaba3f60.0             __dd__Materials__StainlessSteel0xadf7930 
               __dd__Geometry__AD__lvADE--pvSST0xaba3f60.1             __dd__Materials__StainlessSteel0xadf7930 

               http://belle7.nuu.edu.tw/dae/tree/__dd__Geometry__AD__lvADE--pvSST0xaba3f60.0.html 
               http://belle7.nuu.edu.tw/dae/tree/3154.html
                          (4 children, one of which os the Oil)

             child(Oil)-parent(Steel) border

             Thanks to the double ambiguity, this should yield two surfaces ? One for each AD



        [03] <BorderSurface AdDetails__AdSurfacesNear__SSTWaterSurfaceNear1 REFLECTIVITY >
             pv1 (1) Pool__lvNearPoolIWS--pvNearADE10xaa9d608 
               __dd__Geometry__Pool__lvNearPoolIWS--pvNearADE10xaa9d608.0             __dd__Materials__IwsWater0xab82978 

               http://belle7.nuu.edu.tw/dae/tree/__dd__Geometry__Pool__lvNearPoolIWS--pvNearADE10xaa9d608.0.html
               http://belle7.nuu.edu.tw/dae/tree/3153___1.html   cylindrical Iws containing SST

             pv2 (2) AD__lvADE--pvSST0xaba3f60 
               __dd__Geometry__AD__lvADE--pvSST0xaba3f60.0             __dd__Materials__StainlessSteel0xadf7930 
               __dd__Geometry__AD__lvADE--pvSST0xaba3f60.1             __dd__Materials__StainlessSteel0xadf7930 

               http://belle7.nuu.edu.tw/dae/tree/__dd__Geometry__AD__lvADE--pvSST0xaba3f60.0.html
               http://belle7.nuu.edu.tw/dae/tree/3154.html
               http://belle7.nuu.edu.tw/dae/tree/__dd__Geometry__AD__lvADE--pvSST0xaba3f60.1.html
               http://belle7.nuu.edu.tw/dae/tree/4814.html

               Parent(water)-child(Steel), 

        [04] <BorderSurface AdDetails__AdSurfacesNear__SSTWaterSurfaceNear2 REFLECTIVITY >
             pv1 (1) Pool__lvNearPoolIWS--pvNearADE20xaaa18b8 
               __dd__Geometry__Pool__lvNearPoolIWS--pvNearADE20xaaa18b8.0             __dd__Materials__IwsWater0xab82978 

             pv2 (2) AD__lvADE--pvSST0xaba3f60 
               __dd__Geometry__AD__lvADE--pvSST0xaba3f60.0             __dd__Materials__StainlessSteel0xadf7930 
               __dd__Geometry__AD__lvADE--pvSST0xaba3f60.1             __dd__Materials__StainlessSteel0xadf7930 

             Same for other AD, no ambiguity for pv1 but is for pv2


        [05] <BorderSurface PoolDetails__NearPoolSurfaces__NearIWSCurtainSurface BACKSCATTERCONSTANT,SPECULARSPIKECONSTANT,REFLECTIVITY,SPECULARLOBECONSTANT >
             pv1 (1) Pool__lvNearPoolCurtain--pvNearPoolIWS0xae08fa0 
               __dd__Geometry__Pool__lvNearPoolCurtain--pvNearPoolIWS0xae08fa0.0             __dd__Materials__IwsWater0xab82978 

               http://belle7.nuu.edu.tw/dae/tree/__dd__Geometry__Pool__lvNearPoolCurtain--pvNearPoolIWS0xae08fa0.0.html
               http://belle7.nuu.edu.tw/dae/tree/3152.html


             pv2 (1) Pool__lvNearPoolOWS--pvNearPoolCurtain0xae9ba38 
               __dd__Geometry__Pool__lvNearPoolOWS--pvNearPoolCurtain0xae9ba38.0             __dd__Materials__Tyvek0xab26538 

               http://belle7.nuu.edu.tw/dae/tree/__dd__Geometry__Pool__lvNearPoolOWS--pvNearPoolCurtain0xae9ba38.0.html
               http://belle7.nuu.edu.tw/dae/tree/3151.html

               child-parent



        [06] <BorderSurface PoolDetails__NearPoolSurfaces__NearOWSLinerSurface BACKSCATTERCONSTANT,SPECULARSPIKECONSTANT,REFLECTIVITY,SPECULARLOBECONSTANT >
             pv1 (1) Pool__lvNearPoolLiner--pvNearPoolOWS0xaa64f68 
               __dd__Geometry__Pool__lvNearPoolLiner--pvNearPoolOWS0xaa64f68.0             __dd__Materials__OwsWater0xabb2118 

               http://belle7.nuu.edu.tw/dae/tree/__dd__Geometry__Pool__lvNearPoolLiner--pvNearPoolOWS0xaa64f68.0.html
               http://belle7.nuu.edu.tw/dae/tree/3150.html

             pv2 (1) Pool__lvNearPoolDead--pvNearPoolLiner0xab6b300 
               __dd__Geometry__Pool__lvNearPoolDead--pvNearPoolLiner0xab6b300.0             __dd__Materials__Tyvek0xab26538 

               http://belle7.nuu.edu.tw/dae/tree/__dd__Geometry__Pool__lvNearPoolDead--pvNearPoolLiner0xab6b300.0.html
               http://belle7.nuu.edu.tw/dae/tree/3149.html

               child-parent 


        [07] <BorderSurface PoolDetails__NearPoolSurfaces__NearDeadLinerSurface BACKSCATTERCONSTANT,SPECULARSPIKECONSTANT,REFLECTIVITY,SPECULARLOBECONSTANT >

             pv1 (1) Sites__lvNearHallBot--pvNearPoolDead0xaa63ff0 
               __dd__Geometry__Sites__lvNearHallBot--pvNearPoolDead0xaa63ff0.0             __dd__Materials__DeadWater0xaabb308 

               http://belle7.nuu.edu.tw/dae/tree/__dd__Geometry__Sites__lvNearHallBot--pvNearPoolDead0xaa63ff0.0.html
               http://belle7.nuu.edu.tw/dae/tree/3148.html

             pv2 (1) Pool__lvNearPoolDead--pvNearPoolLiner0xab6b300 
               __dd__Geometry__Pool__lvNearPoolDead--pvNearPoolLiner0xab6b300.0             __dd__Materials__Tyvek0xab26538 

               http://belle7.nuu.edu.tw/dae/tree/__dd__Geometry__Pool__lvNearPoolDead--pvNearPoolLiner0xab6b300.0.html
               http://belle7.nuu.edu.tw/dae/tree/3149.html

             parent-child    



How deep does the ambiguity bug go ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. GDML appending the pointer to volume IDs is a crutch, that assumes C++ instance identity 
   and PV identity are equivalent : this issue seems to indicates that is not true





Check VMRL code 
~~~~~~~~~~~~~~~~~

`G4VRML2SceneHandlerFunc.icc`::

    169 void G4VRML2SCENEHANDLER::AddPrimitive(const G4Polyhedron& polyhedron)
    170 { 
    ...
    182     // Current Model
    183     const G4VModel* pv_model  = GetModel();
    184     G4String pv_name = "No model";
    185         if (pv_model) pv_name = pv_model->GetCurrentTag() ;
    186 
    187     // VRML codes are generated below
    188 
    189     //std::cerr << "SCB " << pv_name << "\n";
    190     fDest << "#---------- SOLID: " << pv_name << "\n";
    191 
    192 


`visualization/modeling/include/G4VModel.hh`::

     74   virtual G4String GetCurrentTag () const;
     75   // A tag which depends on the current state of the model.
     76 

`visualization/modeling/src/G4VModel.cc`::

     49 G4String G4VModel::GetCurrentTag () const {
     50   // Override in concrete class if concept of "current" is meaningful.
     51   return fGlobalTag;
     52 }

`visualization/modeling/src/G4PhysicalVolumeModel.cc`::

    181 G4String G4PhysicalVolumeModel::GetCurrentTag () const
    182 {
    183   if (fpCurrentPV) {
    184     std::ostringstream o;
    185     o << fpCurrentPV -> GetCopyNo ();
    186     return fpCurrentPV -> GetName () + "." + o.str();
    187   }
    188   else {
    189     return "WARNING: NO CURRENT VOLUME - global tag is " + fGlobalTag;
    190   }
    191 }
     

PV CopyNo
~~~~~~~~~~~

 
`geometry/management/include/G4VPhysicalVolume.hh`::

    140     virtual G4bool IsMany() const = 0;
    141       // Return true if the volume is MANY (not implemented yet).
    142     virtual G4int GetCopyNo() const = 0;
    143       // Return the volumes copy number.
    144     virtual void  SetCopyNo(G4int CopyNo) = 0;
    145       // Set the volumes copy number.
    146     virtual G4bool IsReplicated() const = 0;
    147       // Return true if replicated (single object instance represents
    148       // many real volumes), else false.
    149     virtual G4bool IsParameterised() const = 0;
    150       // Return true if parameterised (single object instance represents
    151       // many real parameterised volumes), else false.
        

`geometry/volumes/src/G4PVPlacement.cc`::

    174 // GetCopyNo
    175 //
    176 G4int G4PVPlacement::GetCopyNo() const
    177 {
    178   return fcopyNo;
    179 }
    180 
    181 // ----------------------------------------------------------------------
    182 // SetCopyNo
    183 //
    184 void G4PVPlacement::SetCopyNo(G4int newCopyNo)
    185 {
    186   fcopyNo= newCopyNo;
    187 }
    188 


What is setting the CopyNo?::

    [blyth@belle7 source]$ find . -name '*.cc' -exec grep -H SetCopyNo {} \;
    ./persistency/ascii/src/G4tgbPlaceParamCircle.cc:  physVol->SetCopyNo( copyNo );
    ./persistency/ascii/src/G4tgbPlaceParamLinear.cc:  physVol->SetCopyNo( copyNo );
    ./persistency/ascii/src/G4tgbPlaceParamSquare.cc:  physVol->SetCopyNo( copyNo );
    ./visualization/modeling/src/G4PhysicalVolumeModel.cc:  pVPV -> SetCopyNo (n);
    ./visualization/modeling/src/G4PhysicalVolumeModel.cc:  pVPV -> SetCopyNo (n);
    ./geometry/volumes/src/G4PVReplica.cc:void  G4PVReplica::SetCopyNo(G4int newCopyNo)
    ./geometry/volumes/src/G4PVPlacement.cc:// SetCopyNo
    ./geometry/volumes/src/G4PVPlacement.cc:void G4PVPlacement::SetCopyNo(G4int newCopyNo)
    ./geometry/divisions/src/G4PVDivision.cc:void  G4PVDivision::SetCopyNo(G4int newCopyNo)
    ./geometry/navigation/src/G4RegularNavigation.cc:    pPhysical->SetCopyNo(replicaNo);
    ./geometry/navigation/src/G4ParameterisedNavigation.cc:        pPhysical->SetCopyNo(replicaNo);
    ./geometry/navigation/src/G4Navigator.cc:              fBlockedPhysicalVolume->SetCopyNo(fBlockedReplicaNo);
    ./geometry/navigation/src/G4Navigator.cc:                fBlockedPhysicalVolume->SetCopyNo(fBlockedReplicaNo);
    [blyth@belle7 source]$ 



DAE CopyNo
~~~~~~~~~~~

CopyNo is non trivial to persist into DAE, as DAE retains the tree structure unlike VRML2 that fully unwinds it.
The copyNo kinda emerges from the traverse. Despite this it is included in DAE metadata elements, but difficult
to interpret.




