Surfaces Roundtrip
======================

.. contents:: :local:

Info flow
---------

#. NuWa-trunk/dybgaudi/Detector/XmlDetDesc/DDDB detdesc into Geant4 objects via GiGa
#. G4DAE persisted from Geant4 memory into DAE extras
#. ColladaToChroma : pycollada read of extras via DAENode DAEGeometry into Chroma objects
#. Chroma push to GPU


Issue : Model mismatch
------------------------

Sensdet Geant4/NuWa vs Chroma model difference

#. Chroma is expecting sensitive detectors to have associated surfaces
   with all the requisite props to furnish SURFACE_DETECT **hits**.

#. Geant4/NuWa expecting sensitive volumes to...
 

Issue : Missing bordersurface 
--------------------------------

#. All 8 bordersurface are missing 

::

    {'border': ['ESRAirSurfaceTop',
                'ESRAirSurfaceBot',
                'NearIWSCurtain',
                'NearOWSLiner',
                'NearDeadLiner',
                'SSTWaterSurfaceNear1',
                'SSTWaterSurfaceNear2',
                'SSTOil'],


#. pv pairing ambiguity is apparent


#. also need to add 2nd surface index to chroma model 
   and pick which one based on in-to-out or out-to-in

::

    In [9]: bordersurface['SSTOil']    # inside border of SST and MineralOil it contains
    Out[9]: 
    <BorderSurface AdDetails__AdSurfacesAll__SSTOilSurface <OpticalSurface f3 m1 t0 v1 pREFLECTIVITY:4 > >
         pv1 (2) AD__lvSST--pvOIL0xc241510 
           __dd__Geometry__AD__lvSST--pvOIL0xc241510.0             __dd__Materials__MineralOil0xbf5c830 
           __dd__Geometry__AD__lvSST--pvOIL0xc241510.1             __dd__Materials__MineralOil0xbf5c830 
         pv2 (2) AD__lvADE--pvSST0xc128d90 
           __dd__Geometry__AD__lvADE--pvSST0xc128d90.0             __dd__Materials__StainlessSteel0xc2adc00 
           __dd__Geometry__AD__lvADE--pvSST0xc128d90.1             __dd__Materials__StainlessSteel0xc2adc00 


    In [10]: bordersurface['SSTWaterSurfaceNear1']   # outside of SST and surrounding water
    Out[10]: 
    <BorderSurface AdDetails__AdSurfacesNear__SSTWaterSurfaceNear1 <OpticalSurface f3 m1 t0 v1 pREFLECTIVITY:4 > >
         pv1 (1) Pool__lvNearPoolIWS--pvNearADE10xc2cf528 
           __dd__Geometry__Pool__lvNearPoolIWS--pvNearADE10xc2cf528.0             __dd__Materials__IwsWater0xc288f98 
         pv2 (2) AD__lvADE--pvSST0xc128d90 
           __dd__Geometry__AD__lvADE--pvSST0xc128d90.0             __dd__Materials__StainlessSteel0xc2adc00 
           __dd__Geometry__AD__lvADE--pvSST0xc128d90.1             __dd__Materials__StainlessSteel0xc2adc00 

    In [11]: bordersurface['SSTWaterSurfaceNear2']   # huh 
    Out[11]: 
    <BorderSurface AdDetails__AdSurfacesNear__SSTWaterSurfaceNear2 <OpticalSurface f3 m1 t0 v1 pREFLECTIVITY:4 > >
         pv1 (1) Pool__lvNearPoolIWS--pvNearADE20xc0479c8 
           __dd__Geometry__Pool__lvNearPoolIWS--pvNearADE20xc0479c8.0             __dd__Materials__IwsWater0xc288f98 
         pv2 (2) AD__lvADE--pvSST0xc128d90 
           __dd__Geometry__AD__lvADE--pvSST0xc128d90.0             __dd__Materials__StainlessSteel0xc2adc00 
           __dd__Geometry__AD__lvADE--pvSST0xc128d90.1             __dd__Materials__StainlessSteel0xc2adc00 




Seeing how event number 1 photons gets propagated with partial geometry is interesting::

    g4daeview.sh --geometry-regexp ADE1 --load 1                     ## not many photons escape the water ?
    g4daeview.sh --geometry-regexp pvNearADE10xc2cf528.0 --load 1    ## seemingly no photons escape the water column

    g4daeview.sh --geometry-regexp ADE2 --load 1                     ## backspill 
    g4daeview.sh --geometry-regexp pvNearADE20xc0479c8.0  --load 1   ## lots of backspill manage to enter water though  




     



 
detdesc level
~~~~~~~~~~~~~

* perusing DDDB would suggest some surfaces are missing

xmldae level (raw DAE XML)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Comparing ChromaSurfaceMap and raw xml with `xmldae.sh --ipy --surface`  missing marked with `-`::

    delta:~ blyth$ xmldae.sh --ipy --surface
    ...
    [ 11] __dd__Geometry__PoolDetails__NearPoolSurfaces__   

         +NearInnOutPiper
         -NearPoolCover
         +TopShortCableTray
         +NearInnInPiper
         +NearOutOutPiper
         -NearIWSCurtain
         -NearOWSLiner
         -NearDeadLiner
         +UnistrutRib7
         +NearOutInPiper
         +UnistrutRib6

    [  5] __dd__Geometry__AdDetails__AdSurfacesAll__        

         +AdCableTray
         -ESRAirSurfaceTop
         -ESRAirSurfaceBot
         +RSOil
         -SSTOil

    [  2] __dd__Geometry__AdDetails__AdSurfacesNear__       

         -SSTWaterSurfaceNear1
         -SSTWaterSurfaceNear2

    [ 24] __dd__Geometry__PoolDetails__PoolSurfacesAll__    

         +PmtMtRib2
         +TopCornerCableTray
         +TablePanel
         +UnistrutRib2
         +UnistrutRib3
         +UnistrutRib8
         +LegInOWSTub
         +SlopeRib5
         +PmtMtBaseRing
         +LegInDeadTub
         +PmtMtRib1
         +SupportRib5
         +PmtMtRib3
         +UnistrutRib1
         +LegInIWSTub
         +ADVertiCableTray
         +UnistrutRib9
         +VertiCableTray
         +ShortParCableTray
         +SlopeRib1
         +PmtMtTopRing
         +SupportRib1
         +UnistrutRib4
         +UnistrutRib5

    2014-07-11 17:03:04,859 env.geant4.geometry.collada.xmldae INFO     found 42 raw opticalsurface elements, 9 of which are missing from csm 


Initially all 8 bordersurface are missing and 1 skinsurface is missing ('NearPoolCover').
Maybe NearPoolCover missing from CSM as default geometry selection excludes it::

    In [6]: print ET.tostring(raw['skin']['NearPoolCover'])
    <skinsurface xmlns="http://www.collada.org/2005/11/COLLADASchema" name="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearPoolCoverSurface" surfaceproperty="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearPoolCoverSurface">
            <volumeref ref="__dd__Geometry__PoolDetails__lvNearTopCover0xc137060"/>
          </skinsurface>

Yep, after adding a more complete **dybf** geometry and using that get::

    not_in_csm:
    {'border': ['ESRAirSurfaceTop',
                'ESRAirSurfaceBot',
                'NearIWSCurtain',
                'NearOWSLiner',
                'NearDeadLiner',
                'SSTWaterSurfaceNear1',
                'SSTWaterSurfaceNear2',
                'SSTOil'],
     'optical': ['ESRAirSurfaceTop',
                 'ESRAirSurfaceBot',
                 'NearIWSCurtain',
                 'NearOWSLiner',
                 'NearDeadLiner',
                 'SSTWaterSurfaceNear1',
                 'SSTWaterSurfaceNear2',
                 'SSTOil'],
     'skin': []}


    In [3]: not_in_csm['border'] == not_in_csm['optical']
    Out[3]: True





daenode level
~~~~~~~~~~~~~~~~

::

    delta:~ blyth$ daenode.sh --ipy
    2014-07-11 17:38:53,158 env.geant4.geometry.collada.daenode:1763 INFO     /Users/blyth/env/bin/daenode.py
    2014-07-11 17:38:54,914 env.geant4.geometry.collada.daenode:1806 INFO     drop into embedded IPython
    Python 2.7.6 (default, Nov 18 2013, 15:12:51) 
    Type "copyright", "credits" or "license" for more information.

    IPython 1.2.1 -- An enhanced Interactive Python.
    ?         -> Introduction and overview of IPython's features.
    %quickref -> Quick reference.
    help      -> Python's own help system.
    object?   -> Details about 'object', use 'object??' for extra details.

    In [1]: DAENode.extra
    Out[1]: DAEExtra skinsurface 34 bordersurface 8 opticalsurface 42 skinmap 34 bordermap 13 

    In [2]: DAENode.extra.__class__
    Out[2]: env.geant4.geometry.collada.daenode.DAEExtra


    In [2]: DAENode.extra.bordersurface
    Out[2]: 
    [<BorderSurface AdDetails__AdSurfacesAll__ESRAirSurfaceTop <OpticalSurface f0 m1 t0 v0 pREFLECTIVITY:31 > >
         pv1 (2) AdDetails__lvTopReflector--pvTopRefGap0xc266468 
           __dd__Geometry__AdDetails__lvTopReflector--pvTopRefGap0xc266468.0             __dd__Materials__Air0xc032550 
           __dd__Geometry__AdDetails__lvTopReflector--pvTopRefGap0xc266468.1             __dd__Materials__Air0xc032550 
         pv2 (2) AdDetails__lvTopRefGap--pvTopESR0xc4110d0 
           __dd__Geometry__AdDetails__lvTopRefGap--pvTopESR0xc4110d0.0             __dd__Materials__ESR0xbf9f438 
           __dd__Geometry__AdDetails__lvTopRefGap--pvTopESR0xc4110d0.1             __dd__Materials__ESR0xbf9f438 ,
     <BorderSurface AdDetails__AdSurfacesAll__ESRAirSurfaceBot <OpticalSurface f0 m1 t0 v0 pREFLECTIVITY:31 > >
         pv1 (2) AdDetails__lvBotReflector--pvBotRefGap0xbfa6458 
           __dd__Geometry__AdDetails__lvBotReflector--pvBotRefGap0xbfa6458.0             __dd__Materials__Air0xc032550 
           __dd__Geometry__AdDetails__lvBotReflector--pvBotRefGap0xbfa6458.1             __dd__Materials__Air0xc032550 
         pv2 (2) AdDetails__lvBotRefGap--pvBotESR0xbf9bd08 
           __dd__Geometry__AdDetails__lvBotRefGap--pvBotESR0xbf9bd08.0             __dd__Materials__ESR0xbf9f438 
           __dd__Geometry__AdDetails__lvBotRefGap--pvBotESR0xbf9bd08.1             __dd__Materials__ESR0xbf9f438 ,
     <BorderSurface AdDetails__AdSurfacesAll__SSTOilSurface <OpticalSurface f3 m1 t0 v1 pREFLECTIVITY:4 > >
         pv1 (2) AD__lvSST--pvOIL0xc241510 
           __dd__Geometry__AD__lvSST--pvOIL0xc241510.0             __dd__Materials__MineralOil0xbf5c830 
           __dd__Geometry__AD__lvSST--pvOIL0xc241510.1             __dd__Materials__MineralOil0xbf5c830 
         pv2 (2) AD__lvADE--pvSST0xc128d90 
           __dd__Geometry__AD__lvADE--pvSST0xc128d90.0             __dd__Materials__StainlessSteel0xc2adc00 
           __dd__Geometry__AD__lvADE--pvSST0xc128d90.1             __dd__Materials__StainlessSteel0xc2adc00 ,


    In [9]: bordersurface['SSTOil']
    Out[9]: 
    <BorderSurface AdDetails__AdSurfacesAll__SSTOilSurface <OpticalSurface f3 m1 t0 v1 pREFLECTIVITY:4 > >
         pv1 (2) AD__lvSST--pvOIL0xc241510 
           __dd__Geometry__AD__lvSST--pvOIL0xc241510.0             __dd__Materials__MineralOil0xbf5c830 
           __dd__Geometry__AD__lvSST--pvOIL0xc241510.1             __dd__Materials__MineralOil0xbf5c830 
         pv2 (2) AD__lvADE--pvSST0xc128d90 
           __dd__Geometry__AD__lvADE--pvSST0xc128d90.0             __dd__Materials__StainlessSteel0xc2adc00 
           __dd__Geometry__AD__lvADE--pvSST0xc128d90.1             __dd__Materials__StainlessSteel0xc2adc00 

    In [10]: bordersurface['SSTWaterSurfaceNear1']
    Out[10]: 
    <BorderSurface AdDetails__AdSurfacesNear__SSTWaterSurfaceNear1 <OpticalSurface f3 m1 t0 v1 pREFLECTIVITY:4 > >
         pv1 (1) Pool__lvNearPoolIWS--pvNearADE10xc2cf528 
           __dd__Geometry__Pool__lvNearPoolIWS--pvNearADE10xc2cf528.0             __dd__Materials__IwsWater0xc288f98 
         pv2 (2) AD__lvADE--pvSST0xc128d90 
           __dd__Geometry__AD__lvADE--pvSST0xc128d90.0             __dd__Materials__StainlessSteel0xc2adc00 
           __dd__Geometry__AD__lvADE--pvSST0xc128d90.1             __dd__Materials__StainlessSteel0xc2adc00 

    In [11]: bordersurface['SSTWaterSurfaceNear2']
    Out[11]: 
    <BorderSurface AdDetails__AdSurfacesNear__SSTWaterSurfaceNear2 <OpticalSurface f3 m1 t0 v1 pREFLECTIVITY:4 > >
         pv1 (1) Pool__lvNearPoolIWS--pvNearADE20xc0479c8 
           __dd__Geometry__Pool__lvNearPoolIWS--pvNearADE20xc0479c8.0             __dd__Materials__IwsWater0xc288f98 
         pv2 (2) AD__lvADE--pvSST0xc128d90 
           __dd__Geometry__AD__lvADE--pvSST0xc128d90.0             __dd__Materials__StainlessSteel0xc2adc00 
           __dd__Geometry__AD__lvADE--pvSST0xc128d90.1             __dd__Materials__StainlessSteel0xc2adc00 














Use of surfaces by Chroma
---------------------------

Without associated surface_index never get SURFACE_ABSORB, SURFACE_DETECT, REFLECT_DIFFUSE?

`chroma/cuda/propagate_vbo.cu`::

    616        // PASS goes on to either at_surface/at_boundary handling 
    617        // depending on surface association 
    621        if (s.surface_index != -1)
    622        {
    625            // #. SURFACE_ABSORB(BREAK)
    626            // #. SURFACE_DETECT(BREAK)
    627            // #. REFLECT_DIFFUSE(CONTINUE) .direction .polarization
    628            // #. REFLECT_SPECULAR(CONTINUE) .direction
    629            // #. NO other option, so never PASS? 
    631            command = propagate_at_surface(p, s, rng, g, use_weights);
    636            if (command == BREAK) break ;
    637            if (command == CONTINUE) continue;
    641        }
    642 
    643        // **propagate_at_boundary** 
    644        // depending on materials refractive indices and incidence angle
    645        //
    646        // #. "CONTINUE" REFLECT_SPECULAR  .direction .polarization
    647        // #. "CONTINUE" "REFRACT"         .direction .polarization
    648        // #. NO other option
    649 
    650        propagate_at_boundary(p, s, rng);
    654 


surface association
~~~~~~~~~~~~~~~~~~~~~~

Triangle material_codes lookup: `g->material_codes[p.last_hit_triangle]` 
gives materials and maybe surface associated.

::

    165 __device__ void
    166 fill_state(State &s, Photon &p, Geometry *g)
    167 {
    168     p.last_hit_triangle = intersect_mesh(p.position, p.direction, g,
    169                                          s.distance_to_boundary,
    170                                          p.last_hit_triangle);
    ...
    179     Triangle t = get_triangle(g, p.last_hit_triangle);
    180 
    181     unsigned int material_code = g->material_codes[p.last_hit_triangle];
    182 
    183     int inner_material_index = convert(0xFF & (material_code >> 24));
    184     int outer_material_index = convert(0xFF & (material_code >> 16));
    185     s.surface_index = convert(0xFF & (material_code >> 8));
    186 
    ...    convert just makes -1 if needed


propagate_at_surface
~~~~~~~~~~~~~~~~~~~~~~

Wavelength dependant surface props needed by `propagate_at_surface`

* detect
* absorb
* reflect_diffuse
* reflect_specular 


`chroma/cuda/photon.h`::

    701 __device__ int
    702 propagate_at_surface(Photon &p, State &s, curandState &rng, Geometry *geometry,
    703                      bool use_weights=false)
    704 {
    705     Surface *surface = geometry->surfaces[s.surface_index];
    706 
    707     if (surface->model == SURFACE_COMPLEX)
    708         return propagate_complex(p, s, rng, surface, use_weights);
    709     else if (surface->model == SURFACE_WLS)
    710         return propagate_at_wls(p, s, rng, surface, use_weights);
    711     else
    712     {
    713         // use default surface model: do a combination of specular and
    714         // diffuse reflection, detection, and absorption based on relative
    715         // probabilties
    716 
    717         // since the surface properties are interpolated linearly, we are
    718         // guaranteed that they still sum to 1.0.
    719         float detect = interp_property(surface, p.wavelength, surface->detect);
    720         float absorb = interp_property(surface, p.wavelength, surface->absorb);
    721         float reflect_diffuse = interp_property(surface, p.wavelength, surface->reflect_diffuse);
    722         float reflect_specular = interp_property(surface, p.wavelength, surface->reflect_specular);
    723 







