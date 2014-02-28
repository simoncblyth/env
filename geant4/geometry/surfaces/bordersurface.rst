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








