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



Geant4 Usage of  G4LogicalBorderSurface 
-------------------------------------------

::

    [blyth@belle7 source]$ find . -name '*.cc' -exec grep -l G4LogicalBorderSurface {} \;
    ./persistency/gdml/src/G4GDMLReadStructure.cc
    ./processes/optical/src/G4OpBoundaryProcess.cc
    ./geometry/volumes/src/G4LogicalBorderSurface.cc
    [blyth@belle7 source]$ 


`processes/optical/src/G4OpBoundaryProcess.cc`::

    132 G4VParticleChange*
    133 G4OpBoundaryProcess::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep)
    134 {

    218         theModel = glisur;
    219         theFinish = polished;
    220 
    221         G4SurfaceType type = dielectric_dielectric;
    222 
    223         Rindex = NULL;
    224         OpticalSurface = NULL;
    225 
    226         G4LogicalSurface* Surface = NULL;
    227 
    228         Surface = G4LogicalBorderSurface::GetSurface
    229               (pPreStepPoint ->GetPhysicalVolume(),
    230                pPostStepPoint->GetPhysicalVolume());

               // THIS IS ONLY CHANCE TO SNAG A BORDERSURFACE
    231 
    232         if (Surface == NULL){
    233       G4bool enteredDaughter=(pPostStepPoint->GetPhysicalVolume()
    234                   ->GetMotherLogical() ==
    235                   pPreStepPoint->GetPhysicalVolume()
    236                   ->GetLogicalVolume());

              //  poststep.PV.motherLV == prestep.PV.LV
              //  poststep mother logical is prestep logical ... so poststep must be daughter of prestep  

    237       if(enteredDaughter){             
    238         Surface = G4LogicalSkinSurface::GetSurface
    239           (pPostStepPoint->GetPhysicalVolume()->
    240            GetLogicalVolume());

    241         if(Surface == NULL)
    242           Surface = G4LogicalSkinSurface::GetSurface
    243           (pPreStepPoint->GetPhysicalVolume()->
    244            GetLogicalVolume());
    245       }
    246       else {

             // does this imply poststep must be mother of prestep ?   
             // what about stepping between siblings ?

    247         Surface = G4LogicalSkinSurface::GetSurface
    248           (pPreStepPoint->GetPhysicalVolume()->
    249            GetLogicalVolume());
    250         if(Surface == NULL)
    251           Surface = G4LogicalSkinSurface::GetSurface
    252           (pPostStepPoint->GetPhysicalVolume()->
    253            GetLogicalVolume());
    254       }
    255     }

     Translating that into something digestable, 

             //   Surface = G4LogicalBorderSurface::GetSurface(pPreStepPoint ->GetPhysicalVolume(),pPostStepPoint->GetPhysicalVolume());
             //      *  border surface takes priority 
             //
             //   if(Surface == NULL){ 
             //   ...   
             //   if(enteredDaughter){    // first try post and then pre : daughter has first dibs
             //        Surface = G4LogicalSkinSurface::GetSurface(pPostStepPoint->GetPhysicalVolume()->GetLogicalVolume());
             //        if(Surface == NULL){
             //           Surface = G4LogicalSkinSurface::GetSurface(pPreStepPoint->GetPhysicalVolume()->GetLogicalVolume());
             //         }
             //   }
             //   else
             //   {                // first pre then post : 
             //
             //   




    256 
    257     if (Surface) OpticalSurface =
    258            dynamic_cast <G4OpticalSurface*> (Surface->GetSurfaceProperty());
    259 






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



DAE Debug
------------

Interleaving the bordersurface with the debug bsurf from the meta element. Observations:

* one extra bsurf, 
* copyNo not helping... presumably because of when it is requested, need to do this during the traverse somehow as the copyNo is being incremented

::

    157610       <bordersurface name="__dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceTop" surfaceproperty="__dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceTop">
    157611         <physvolref ref="__dd__Geometry__AdDetails__lvTopReflector--pvTopRefGap0xc05e0f0"/>
    157612         <physvolref ref="__dd__Geometry__AdDetails__lvTopRefGap--pvTopESR0xc208d58"/>
    157613       </bordersurface>
    157614       <bordersurface name="__dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceBot" surfaceproperty="__dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceBot">
    157615         <physvolref ref="__dd__Geometry__AdDetails__lvBotReflector--pvBotRefGap0xbd9e0e0"/>
    157616         <physvolref ref="__dd__Geometry__AdDetails__lvBotRefGap--pvBotESR0xbd93990"/>
    157617       </bordersurface>
    157618       <bordersurface name="__dd__Geometry__AdDetails__AdSurfacesAll__SSTOilSurface" surfaceproperty="__dd__Geometry__AdDetails__AdSurfacesAll__SSTOilSurface">
    157619         <physvolref ref="__dd__Geometry__AD__lvSST--pvOIL0xc039198"/>
    157620         <physvolref ref="__dd__Geometry__AD__lvADE--pvSST0xbf20a18"/>
    157621       </bordersurface>

    157642       <meta>
    157643         <bsurf name="__dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceTop" surfaceproperty="__dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceTop">
    157644           <pv copyNo="1000" name="__dd__Geometry__AdDetails__lvTopReflector--pvTopRefGap" ref="__dd__Geometry__AdDetails__lvTopReflector--pvTopRefGap0xc05e0f0"/>
    157645           <pv copyNo="1000" name="__dd__Geometry__AdDetails__lvTopRefGap--pvTopESR" ref="__dd__Geometry__AdDetails__lvTopRefGap--pvTopESR0xc208d58"/>
    157646         </bsurf>
    157647         <bsurf name="__dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceBot" surfaceproperty="__dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceBot">
    157648           <pv copyNo="1000" name="__dd__Geometry__AdDetails__lvBotReflector--pvBotRefGap" ref="__dd__Geometry__AdDetails__lvBotReflector--pvBotRefGap0xbd9e0e0"/>
    157649           <pv copyNo="1000" name="__dd__Geometry__AdDetails__lvBotRefGap--pvBotESR" ref="__dd__Geometry__AdDetails__lvBotRefGap--pvBotESR0xbd93990"/>
    157650         </bsurf>
    157651         <bsurf name="__dd__Geometry__AdDetails__AdSurfacesAll__SSTOilSurface" surfaceproperty="__dd__Geometry__AdDetails__AdSurfacesAll__SSTOilSurface">
    157652           <pv copyNo="1000" name="__dd__Geometry__AD__lvSST--pvOIL" ref="__dd__Geometry__AD__lvSST--pvOIL0xc039198"/>
    157653           <pv copyNo="1000" name="__dd__Geometry__AD__lvADE--pvSST" ref="__dd__Geometry__AD__lvADE--pvSST0xbf20a18"/>
    157654         </bsurf>




    157622       <bordersurface name="__dd__Geometry__AdDetails__AdSurfacesNear__SSTWaterSurfaceNear1" surfaceproperty="__dd__Geometry__AdDetails__AdSurfacesNear__SSTWaterSurfaceNear1">
    157623         <physvolref ref="__dd__Geometry__Pool__lvNearPoolIWS--pvNearADE10xc0c71b0"/>
    157624         <physvolref ref="__dd__Geometry__AD__lvADE--pvSST0xbf20a18"/>
    157625       </bordersurface>
    157626       <bordersurface name="__dd__Geometry__AdDetails__AdSurfacesNear__SSTWaterSurfaceNear2" surfaceproperty="__dd__Geometry__AdDetails__AdSurfacesNear__SSTWaterSurfaceNear2">
    157627         <physvolref ref="__dd__Geometry__Pool__lvNearPoolIWS--pvNearADE20xbe3f650"/>
    157628         <physvolref ref="__dd__Geometry__AD__lvADE--pvSST0xbf20a18"/>
    157629       </bordersurface>

    157655         <bsurf name="__dd__Geometry__AdDetails__AdSurfacesNear__SSTWaterSurfaceNear1" surfaceproperty="__dd__Geometry__AdDetails__AdSurfacesNear__SSTWaterSurfaceNear1">
    157656           <pv copyNo="1000" name="__dd__Geometry__Pool__lvNearPoolIWS--pvNearADE1" ref="__dd__Geometry__Pool__lvNearPoolIWS--pvNearADE10xc0c71b0"/>
    157657           <pv copyNo="1000" name="__dd__Geometry__AD__lvADE--pvSST" ref="__dd__Geometry__AD__lvADE--pvSST0xbf20a18"/>
    157658         </bsurf>
    157659         <bsurf name="__dd__Geometry__AdDetails__AdSurfacesNear__SSTWaterSurfaceNear2" surfaceproperty="__dd__Geometry__AdDetails__AdSurfacesNear__SSTWaterSurfaceNear2">
    157660           <pv copyNo="1001" name="__dd__Geometry__Pool__lvNearPoolIWS--pvNearADE2" ref="__dd__Geometry__Pool__lvNearPoolIWS--pvNearADE20xbe3f650"/>
    157661           <pv copyNo="1000" name="__dd__Geometry__AD__lvADE--pvSST" ref="__dd__Geometry__AD__lvADE--pvSST0xbf20a18"/>
    157662         </bsurf>




    157630       <bordersurface name="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearIWSCurtainSurface" surfaceproperty="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearIWSCurtainSurface">
    157631         <physvolref ref="__dd__Geometry__Pool__lvNearPoolCurtain--pvNearPoolIWS0xbf52120"/>
    157632         <physvolref ref="__dd__Geometry__Pool__lvNearPoolOWS--pvNearPoolCurtain0xc3bdb90"/>
    157633       </bordersurface>

    157675         <bsurf name="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearIWSCurtainSurface" surfaceproperty="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearIWSCurtainSurface">
    157676           <pv copyNo="1000" name="__dd__Geometry__Pool__lvNearPoolCurtain--pvNearPoolIWS" ref="__dd__Geometry__Pool__lvNearPoolCurtain--pvNearPoolIWS0xbf52120"/>
    157677           <pv copyNo="1000" name="__dd__Geometry__Pool__lvNearPoolOWS--pvNearPoolCurtain" ref="__dd__Geometry__Pool__lvNearPoolOWS--pvNearPoolCurtain0xc3bdb90"/>
    157678         </bsurf>



    157634       <bordersurface name="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearOWSLinerSurface" surfaceproperty="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearOWSLinerSurface">
    157635         <physvolref ref="__dd__Geometry__Pool__lvNearPoolLiner--pvNearPoolOWS0xbd579a8"/>
    157636         <physvolref ref="__dd__Geometry__Pool__lvNearPoolDead--pvNearPoolLiner0xbd42ef8"/>
    157637       </bordersurface>

    157663         <bsurf name="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearOWSLinerSurface" surfaceproperty="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearOWSLinerSurface">
    157664           <pv copyNo="1000" name="__dd__Geometry__Pool__lvNearPoolLiner--pvNearPoolOWS" ref="__dd__Geometry__Pool__lvNearPoolLiner--pvNearPoolOWS0xbd579a8"/>
    157665           <pv copyNo="1000" name="__dd__Geometry__Pool__lvNearPoolDead--pvNearPoolLiner" ref="__dd__Geometry__Pool__lvNearPoolDead--pvNearPoolLiner0xbd42ef8"/>
    157666         </bsurf>

       ########### HUH THIS ONE WITH SAME PV1 AS PRIOR IS MISSING IN THE ABOVE
       ########### THE GDML EXTRACTED CODE THAT JUST CHECKS THE FIRST OF THE PAIR IS WRONG
          
    157671         <bsurf name="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearOWSCurtainSurface" surfaceproperty="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearOWSCurtainSurface">
    157672           <pv copyNo="1000" name="__dd__Geometry__Pool__lvNearPoolLiner--pvNearPoolOWS" ref="__dd__Geometry__Pool__lvNearPoolLiner--pvNearPoolOWS0xbd579a8"/>
    157673           <pv copyNo="1000" name="__dd__Geometry__Pool__lvNearPoolOWS--pvNearPoolCurtain" ref="__dd__Geometry__Pool__lvNearPoolOWS--pvNearPoolCurtain0xc3bdb90"/>
    157674         </bsurf>




    157638       <bordersurface name="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearDeadLinerSurface" surfaceproperty="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearDeadLinerSurface">
    157639         <physvolref ref="__dd__Geometry__Sites__lvNearHallBot--pvNearPoolDead0xbf33ca0"/>
    157640         <physvolref ref="__dd__Geometry__Pool__lvNearPoolDead--pvNearPoolLiner0xbd42ef8"/>
    157641       </bordersurface>

    157667         <bsurf name="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearDeadLinerSurface" surfaceproperty="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearDeadLinerSurface">
    157668           <pv copyNo="1000" name="__dd__Geometry__Sites__lvNearHallBot--pvNearPoolDead" ref="__dd__Geometry__Sites__lvNearHallBot--pvNearPoolDead0xbf33ca0"/>
    157669           <pv copyNo="1000" name="__dd__Geometry__Pool__lvNearPoolDead--pvNearPoolLiner" ref="__dd__Geometry__Pool__lvNearPoolDead--pvNearPoolLiner0xbd42ef8"/>
    157670         </bsurf>




    157679       </meta>
    157680     </extra>
    157681   </library_nodes>



GiGa Creation
--------------

::

    [blyth@belle7 lhcb]$ find . -name '*.cpp' -exec grep -H G4LogicalBorderSurface {} \;
    ./Sim/GiGaCnv/src/component/GiGaSurfaceCnv.cpp:#include  "G4LogicalBorderSurface.hh"
    ./Sim/GiGaCnv/src/component/GiGaSurfaceCnv.cpp:  G4LogicalBorderSurface* surf = 
    ./Sim/GiGaCnv/src/component/GiGaSurfaceCnv.cpp:    G4LogicalBorderSurface::GetSurface( pv1 , pv2 );
    ./Sim/GiGaCnv/src/component/GiGaSurfaceCnv.cpp:    { logsurf = new G4LogicalBorderSurface( surface->registry()->identifier() , 
    [blyth@belle7 lhcb]$ 



Ambiguity By Design ? Detdesc XML would suggest so
-----------------------------------------------------

* see geant4/geometry/materials/dd.py for detdesc parsing

::

    [blyth@belle7 DDDB]$ pwd
    /data1/env/local/dybx/NuWa-trunk/dybgaudi/Detector/XmlDetDesc/DDDB

    [blyth@belle7 DDDB]$ find . -name surfaces.xml
    ./AdDetails/surfaces.xml
    ./Parameters/surfaces.xml
    ./AdDetails_213/surfaces.xml
    ./PoolDetails/surfaces.xml


Only one surface definition for all such reflectors in all ADs ?

AdDetails/surfaces.xml::

     43 
     44   <!-- Reflector top and bottom -->
     45 
     46   <surface name="ESRAirSurfaceTop"
     47        model="unified"
     48        finish="polished"
     49        type="dielectric_metal"
     50        value="0.0"
     51        volfirst="/dd/Geometry/AdDetails/lvTopReflector#pvTopRefGap"
     52        volsecond="/dd/Geometry/AdDetails/lvTopRefGap#pvTopESR">
     53     <tabprops address="/dd/Geometry/AdDetails/AdTabProperties/ESRAirReflectivity"/>
     54   </surface>
     55   <surface name="ESRAirSurfaceBot"
     56        model="unified"
     57        finish="polished"
     58        type="dielectric_metal"
     59        value="0.0"
     60        volfirst="/dd/Geometry/AdDetails/lvBotReflector#pvBotRefGap"
     61        volsecond="/dd/Geometry/AdDetails/lvBotRefGap#pvBotESR">
     62     <tabprops address="/dd/Geometry/AdDetails/AdTabProperties/ESRAirReflectivity"/>
     63   </surface>


Parameters/surfaces.xml::

     09 <!-- Geant4's G4OpticalSurface enums -->
     10 <parameter name="polished" value="0"/>
     11 <parameter name="polishedfrontpainted" value="1" />
     12 <parameter name="polishedbackpainted" value="2" />
     13 <parameter name="ground" value="3" />
     14 <parameter name="groundfrontpainted" value="4" />
     15 <parameter name="groundbackpainted" value="5" />
     16 
     17 <parameter name="dielectric_metal" value="0" />
     18 <parameter name="dielectric_dielectric" value="1" />
     19 
     20 <parameter name="glisur" value="0" />
     21 <parameter name="unified" value="1" />


::

    [blyth@belle7 DDDB]$ diff AdDetails/surfaces.xml AdDetails_213/surfaces.xml
    6a7,8
    > <!-- Modified for 2-1-3 configuration -->
    > 
    18c20
    <     <surfaceref href="#SSTWaterSurfaceNear2"/>
    ---
    >     <!-- REMOVED surfaceref href="#SSTWaterSurfaceNear2"/ Unneeded for 2-1-3 config -->
    24c26
    <     <surfaceref href="#SSTWaterSurfaceFar4"/>
    ---
    >     <!-- REMOVED surfaceref href="#SSTWaterSurfaceFar4"/  Unneeded for 2-1-3 config -->
    28,37c30,39
    <     <tabpropertyref href="properties.xml#RSOilReflectivity"/> <!--Radial Shield-->
    <     <tabpropertyref href="properties.xml#RSOilSpecularLobe"/> <!--Radial Shield-->
    <     <tabpropertyref href="properties.xml#RSOilSpecularSpike"/> <!--Radial Shield-->
    <     <tabpropertyref href="properties.xml#RSOilBackScattering"/> <!--Radial Shield-->
    <     <tabpropertyref href="properties.xml#ESRAirReflectivity"/>
    <     <tabpropertyref href="properties.xml#ESRAirSpecularLobe"/>
    <     <tabpropertyref href="properties.xml#ESRAirSpecularSpike"/>
    <     <tabpropertyref href="properties.xml#ESRAirBackScattering"/>
    <     <tabpropertyref href="properties.xml#SSTOilReflectivity"/>
    <     <tabpropertyref href="properties.xml#SSTWaterReflectivity"/>
    ---
    >     <tabpropertyref href="../AdDetails/properties.xml#RSOilReflectivity"/> <!--Radial Shield-->
    >     <tabpropertyref href="../AdDetails/properties.xml#RSOilSpecularLobe"/> <!--Radial Shield-->
    >     <tabpropertyref href="../AdDetails/properties.xml#RSOilSpecularSpike"/> <!--Radial Shield-->
    >     <tabpropertyref href="../AdDetails/properties.xml#RSOilBackScattering"/> <!--Radial Shield-->
    >     <tabpropertyref href="../AdDetails/properties.xml#ESRAirReflectivity"/>
    >     <tabpropertyref href="../AdDetails/properties.xml#ESRAirSpecularLobe"/>
    >     <tabpropertyref href="../AdDetails/properties.xml#ESRAirSpecularSpike"/>
    >     <tabpropertyref href="../AdDetails/properties.xml#ESRAirBackScattering"/>
    >     <tabpropertyref href="../AdDetails/properties.xml#SSTOilReflectivity"/>
    >     <tabpropertyref href="../AdDetails/properties.xml#SSTWaterReflectivity"/>
    88,96c90,92
    <   <surface name="SSTWaterSurfaceNear2"
    <          model="unified"
    <          finish="ground"
    <          type="dielectric_metal"
    <          value="1.0"
    <          volfirst="/dd/Geometry/Pool/lvNearPoolIWS#pvNearADE2"
    <          volsecond="/dd/Geometry/AD/lvADE#pvSST">
    <     <tabprops address="/dd/Geometry/AdDetails/AdTabProperties/SSTWaterReflectivity"/>
    <   </surface>
    ---
    > 
    >   <!-- Removed pvNearADE2 for 2-1-3 configuration -->
    > 
    128,136c124,125
    <   <surface name="SSTWaterSurfaceFar4"
    <          model="unified"
    <          finish="ground"
    <          type="dielectric_metal"
    <          value="1.0"
    <          volfirst="/dd/Geometry/Pool/lvFarPoolIWS#pvFarADE4"
    <          volsecond="/dd/Geometry/AD/lvADE#pvSST">
    <     <tabprops address="/dd/Geometry/AdDetails/AdTabProperties/SSTWaterReflectivity"/>
    <   </surface>
    ---
    > 
    >   <!-- Removed pvFarADE4 for 2-1-3 configuration --> 
    [blyth@belle7 DDDB]$ 





BorderSurface Debug during traverse
--------------------------------------

* PV2 matches on the volume before (the mother ?)


::

    G4DAE::GetBorderSurface ... /dd/Geometry/AD/lvOIL#pvRadialShield:27[27]
    G4DAE::GetBorderSurface ... /dd/Geometry/AD/lvOIL#pvRadialShield:28[28]
    G4DAE::GetBorderSurface ... /dd/Geometry/AD/lvOIL#pvRadialShield:29[29]
    G4DAE::GetBorderSurface ... /dd/Geometry/AD/lvOIL#pvRadialShield:30[30]
    G4DAE::GetBorderSurface ... /dd/Geometry/AD/lvOIL#pvRadialShield:31[31]

    G4DAE::GetBorderSurface ... /dd/Geometry/AdDetails/lvTopRefGap#pvTopESR[1000]
    G4DAE::GetBorderSurface surf PV2 match 
             PV1 [copyNo]name [1000]/dd/Geometry/AdDetails/lvTopReflector#pvTopRefGap
             PV2 [copyNo]name [1000]/dd/Geometry/AdDetails/lvTopRefGap#pvTopESR
    G4DAE::GetBorderSurface ... /dd/Geometry/AdDetails/lvTopReflector#pvTopRefGap[1000]
    G4DAE::GetBorderSurface surf_first_pv1 
             PV1 [copyNo]name [1000]/dd/Geometry/AdDetails/lvTopReflector#pvTopRefGap
             PV2 [copyNo]name [1000]/dd/Geometry/AdDetails/lvTopRefGap#pvTopESR
    G4DAE::GetBorderSurface ... /dd/Geometry/AD/lvOIL#pvTopReflector[1429]
    G4DAE::GetBorderSurface ... /dd/Geometry/AdDetails/lvBotRefGap#pvBotESR[1000]
    G4DAE::GetBorderSurface surf PV2 match 
             PV1 [copyNo]name [1000]/dd/Geometry/AdDetails/lvBotReflector#pvBotRefGap
             PV2 [copyNo]name [1000]/dd/Geometry/AdDetails/lvBotRefGap#pvBotESR
    G4DAE::GetBorderSurface ... /dd/Geometry/AdDetails/lvBotReflector#pvBotRefGap[1000]
    G4DAE::GetBorderSurface surf_first_pv1 
             PV1 [copyNo]name [1000]/dd/Geometry/AdDetails/lvBotReflector#pvBotRefGap
             PV2 [copyNo]name [1000]/dd/Geometry/AdDetails/lvBotRefGap#pvBotESR
    G4DAE::GetBorderSurface ... /dd/Geometry/AD/lvOIL#pvBotReflector[1430]
    G4DAE::GetBorderSurface ... /dd/Geometry/AD/lvOIL#pvSstBotRadiusRibs#SstBotRibs#SstBotRibRot[1431]
    G4DAE::GetBorderSurface ... /dd/Geometry/AD/lvOIL#pvSstBotRadiusRibs#SstBotRibs:1#SstBotRibRot[1]
    G4DAE::GetBorderSurface ... /dd/Geometry/AD/lvOIL#pvSstBotRadiusRibs#SstBotRibs:2#SstBotRibRot[2]
    G4DAE::GetBorderSurface ... /dd/Geometry/AD/lvOIL#pvSstBotRadiusRibs#SstBotRibs:3#SstBotRibRot[3]
    


    G4DAE::GetBorderSurface ... /dd/Geometry/AD/lvOIL#pvWallUpperLedSourceAssy[1517]
    G4DAE::GetBorderSurface ... /dd/Geometry/AD/lvOIL#pvWallMidLedSourceAssy[1518]
    G4DAE::GetBorderSurface ... /dd/Geometry/AD/lvOIL#pvWallBotLedSourceAssy[1519]
    G4DAE::GetBorderSurface ... /dd/Geometry/AD/lvSST#pvOIL[1000]
    G4DAE::GetBorderSurface surf_first_pv1 
             PV1 [copyNo]name [1000]/dd/Geometry/AD/lvSST#pvOIL
             PV2 [copyNo]name [1000]/dd/Geometry/AD/lvADE#pvSST
    G4DAE::GetBorderSurface ... /dd/Geometry/AD/lvSST#lvCenterCalibHoleSST[1001]
    G4DAE::GetBorderSurface ... /dd/Geometry/AD/lvSST#pvOffCenterCalibHoleSST[1002]
    G4DAE::GetBorderSurface ... /dd/Geometry/AD/lvSST#pvGCatCalibHoleSST[1003]
    G4DAE::GetBorderSurface ... /dd/Geometry/AD/lvADE#pvSST[1000]
    G4DAE::GetBorderSurface surf PV2 match 
             PV1 [copyNo]name [1000]/dd/Geometry/AD/lvSST#pvOIL
             PV2 [copyNo]name [1000]/dd/Geometry/AD/lvADE#pvSST
    G4DAE::GetBorderSurface surf PV2 match 
             PV1 [copyNo]name [1000]/dd/Geometry/Pool/lvNearPoolIWS#pvNearADE1
             PV2 [copyNo]name [1000]/dd/Geometry/AD/lvADE#pvSST
    G4DAE::GetBorderSurface surf PV2 match 
             PV1 [copyNo]name [1001]/dd/Geometry/Pool/lvNearPoolIWS#pvNearADE2
             PV2 [copyNo]name [1000]/dd/Geometry/AD/lvADE#pvSST
    G4DAE::GetBorderSurface ... /dd/Geometry/CalibrationBox/lvCenterCalibE#pvCenterBottomPlate[1000]
    G4DAE::GetBorderSurface ... /dd/Geometry/CalibrationBox/lvDomeInterior#pvShieldingPuck[1000]
    G4DAE::GetBorderSurface ... /dd/Geometry/CalibrationBox/lvDomeInterior#pvBearingRing[1001]
    G4DAE::GetBorderSurface ... /dd/Geometry/CalibrationBox/lvDomeInterior#pvTurntableLowerPlate[1002]



    G4DAE::GetBorderSurface ... /dd/Geometry/AdDetails/lvMOOverflowTankE#pvMOFTTopFlangeInterior[1004]
    G4DAE::GetBorderSurface ... /dd/Geometry/AdDetails/lvMOOverflowTankE#pvMOFTTopCover[1005]
    G4DAE::GetBorderSurface ... /dd/Geometry/AD/lvADE#pvlvMOOverflowTankE1[1009]
    G4DAE::GetBorderSurface ... /dd/Geometry/AD/lvADE#pvlvMOOverflowTankE2[1010]
    G4DAE::GetBorderSurface ... /dd/Geometry/Pool/lvNearPoolIWS#pvNearADE1[1000]
    G4DAE::GetBorderSurface surf_first_pv1 
             PV1 [copyNo]name [1000]/dd/Geometry/Pool/lvNearPoolIWS#pvNearADE1
             PV2 [copyNo]name [1000]/dd/Geometry/AD/lvADE#pvSST
    G4DAE::GetBorderSurface ... /dd/Geometry/Pool/lvNearPoolIWS#pvNearADE2[1001]
    G4DAE::GetBorderSurface surf_first_pv1 
             PV1 [copyNo]name [1001]/dd/Geometry/Pool/lvNearPoolIWS#pvNearADE2
             PV2 [copyNo]name [1000]/dd/Geometry/AD/lvADE#pvSST
    G4DAE::GetBorderSurface ... /dd/Geometry/Pool/lvNearPoolIWS#pvVetoPmtNearInn#pvNearInnWall1#pvNearInnWall1:1#pvVetoPmtUnit#pvPmtHemi[1]
    G4DAE::GetBorderSurface ... /dd/Geometry/Pool/lvNearPoolIWS#pvVetoPmtNearInn#pvNearInnWall1#pvNearInnWall1:1#pvVetoPmtUnit#pvPmtMount#pvPmtTopRing[1]
    G4DAE::GetBorderSurface ... /dd/Geometry/Pool/lvNearPoolIWS#pvVetoPmtNearInn#pvNearInnWall1#pvNearInnWall1:1#pvVetoPmtUnit#pvPmtMount#pvPmtBaseRing[1]



    G4DAE::GetBorderSurface ... /dd/Geometry/Pool/lvNearPoolOWS#pvNearMuonCableTray#pvMuonCableTrayNear:2#MuonHalfCableTrayNear:7[2]
    G4DAE::GetBorderSurface ... /dd/Geometry/Pool/lvNearPoolOWS#pvOutWaterPipeNear#OutWaterPipeNear:1[1]
    G4DAE::GetBorderSurface ... /dd/Geometry/Pool/lvNearPoolOWS#pvOutWaterPipeNear#OutWaterPipeNear:2[2]
    G4DAE::GetBorderSurface ... /dd/Geometry/Pool/lvNearPoolLiner#pvNearPoolOWS[1000]
    G4DAE::GetBorderSurface surf_first_pv1 
             PV1 [copyNo]name [1000]/dd/Geometry/Pool/lvNearPoolLiner#pvNearPoolOWS
             PV2 [copyNo]name [1000]/dd/Geometry/Pool/lvNearPoolDead#pvNearPoolLiner
    G4DAE::GetBorderSurface surf other PV1 match 
             PV1 [copyNo]name [1000]/dd/Geometry/Pool/lvNearPoolLiner#pvNearPoolOWS
             PV2 [copyNo]name [1000]/dd/Geometry/Pool/lvNearPoolOWS#pvNearPoolCurtain
    G4DAE::GetBorderSurface ... /dd/Geometry/Pool/lvNearPoolLiner#pvNearADE1LinerLegs#pvLegInLiner:1#pvLegInLinerUnit[1]
    G4DAE::GetBorderSurface ... /dd/Geometry/Pool/lvNearPoolLiner#pvNearADE1LinerLegs#pvLegInLiner:2#pvLegInLinerUnit[2]
    G4DAE::GetBorderSurface ... /dd/Geometry/Pool/lvNearPoolLiner#pvNearADE1LinerLegs#pvLegInLiner:3#pvLegInLinerUnit[3]
    G4DAE::GetBorderSurface ... /dd/Geometry/Pool/lvNearPoolLiner#pvNearADE1LinerLegs#pvLegInLiner:4#pvLegInLinerUnit[4]
    G4DAE::GetBorderSurface ... /dd/Geometry/Pool/lvNearPoolLiner#pvNearADE2LinerLegs#pvLegInLiner:1#pvLegInLinerUnit[1]
    G4DAE::GetBorderSurface ... /dd/Geometry/Pool/lvNearPoolLiner#pvNearADE2LinerLegs#pvLegInLiner:2#pvLegInLinerUnit[2]
    G4DAE::GetBorderSurface ... /dd/Geometry/Pool/lvNearPoolLiner#pvNearADE2LinerLegs#pvLegInLiner:3#pvLegInLinerUnit[3]
    G4DAE::GetBorderSurface ... /dd/Geometry/Pool/lvNearPoolLiner#pvNearADE2LinerLegs#pvLegInLiner:4#pvLegInLinerUnit[4]
    G4DAE::GetBorderSurface ... /dd/Geometry/Pool/lvNearPoolDead#pvNearPoolLiner[1000]
    G4DAE::GetBorderSurface surf PV2 match 
             PV1 [copyNo]name [1000]/dd/Geometry/Pool/lvNearPoolLiner#pvNearPoolOWS
             PV2 [copyNo]name [1000]/dd/Geometry/Pool/lvNearPoolDead#pvNearPoolLiner
    G4DAE::GetBorderSurface surf PV2 match 
             PV1 [copyNo]name [1000]/dd/Geometry/Sites/lvNearHallBot#pvNearPoolDead
             PV2 [copyNo]name [1000]/dd/Geometry/Pool/lvNearPoolDead#pvNearPoolLiner
    G4DAE::GetBorderSurface ... /dd/Geometry/Pool/lvNearPoolDead#pvNearADE1DeadLegs#pvLegInDead:1#pvLegInDeadUnit[1]
    G4DAE::GetBorderSurface ... /dd/Geometry/Pool/lvNearPoolDead#pvNearADE1DeadLegs#pvLegInDead:2#pvLegInDeadUnit[2]
    G4DAE::GetBorderSurface ... /dd/Geometry/Pool/lvNearPoolDead#pvNearADE1DeadLegs#pvLegInDead:3#pvLegInDeadUnit[3]
    G4DAE::GetBorderSurface ... /dd/Geometry/Pool/lvNearPoolDead#pvNearADE1DeadLegs#pvLegInDead:4#pvLegInDeadUnit[4]
    G4DAE::GetBorderSurface ... /dd/Geometry/Pool/lvNearPoolDead#pvNearADE2DeadLegs#pvLegInDead:1#pvLegInDeadUnit[1]
    G4DAE::GetBorderSurface ... /dd/Geometry/Pool/lvNearPoolDead#pvNearADE2DeadLegs#pvLegInDead:2#pvLegInDeadUnit[2]
    G4DAE::GetBorderSurface ... /dd/Geometry/Pool/lvNearPoolDead#pvNearADE2DeadLegs#pvLegInDead:3#pvLegInDeadUnit[3]
    G4DAE::GetBorderSurface ... /dd/Geometry/Pool/lvNearPoolDead#pvNearADE2DeadLegs#pvLegInDead:4#pvLegInDeadUnit[4]
    G4DAE::GetBorderSurface ... /dd/Geometry/Sites/lvNearHallBot#pvNearPoolDead[1000]
    G4DAE::GetBorderSurface surf_first_pv1 
             PV1 [copyNo]name [1000]/dd/Geometry/Sites/lvNearHallBot#pvNearPoolDead
             PV2 [copyNo]name [1000]/dd/Geometry/Pool/lvNearPoolDead#pvNearPoolLiner
    G4DAE::GetBorderSurface ... /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab1[1001]
    G4DAE::GetBorderSurface ... /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab2[1002]
    G4DAE::GetBorderSurface ... /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab3[1003]
    G4DAE::GetBorderSurface ... /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab4[1004]
    G4DAE::GetBorderSurface ... /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab5[1005]




