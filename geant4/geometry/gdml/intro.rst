GDML
=====

* http://gdml.web.cern.ch/GDML/
* http://gdml.web.cern.ch/GDML/gdmlexample.html


The Geometry Description Markup Language is an application-indepedent geometry
description format based on XML. It can be used as the primary geometry
implementation language as well as it provides a geometry data exchange format
for the existing applications. The workpackage consists of the GDML Schema
part, which is a fully self-consistent definition of the GDML syntax. Since
release 3_0_0, the GDML I/O part which provides means for writing out and
reading in GDML files is integrated in the packages (like Geant4 and Root)
providing GDML compliant interfaces. The GDML Schema does not depend in any way
on the GDML I/O parts. In particular, new extensions to the GDML I/O packages
can be implemented (interfaces to new application, like for instance
visualisation, editors, etc), while the GDML Schema definition remains
unchanged.  At the present moment, there exist two toolkit bindings for GDML,
the Geant4 binding and the Root binding, both integrated within the respective
frameworks. Both bindings support the GDML import (reading GDML files) as well
as the export (writing out GDML files).


* http://gdml.web.cern.ch/GDML/doc/GDMLmanual.pdf

This manual will focus on the Geant4 binding to GDML, which, starting from release 9.2 of 
the simulation toolkit, it is now integrated in Geant4.


Overviews
-----------

* http://cao-gdml.in2p3.fr/presentations/GDML_orsay.pdf

   * GDML reader/write are part of the Geant4 and ROOT releases
   * volumes can have auxiliary fields for storing any key/value pairs, eg sensdet
   


Extensions
-----------

Linear Collider Detector Description (LCDD)    
extends GDML with Geant4 specific information    
(sensitive detectors, physics cuts, etc)  
   
Full Detector Simulation using SLIC and LCDD  
  * http://www.slac.stanford.edu/cgi-wrap/getdoc/slac-pub-11418.pdf


Geant4 GDML 
------------

* http://geant4.slac.stanford.edu/SLACTutorial09/Geometry3.pdf


Module Examples
~~~~~~~~~~~~~~~~~

::

    [blyth@cms01 geant4.9.2.p01]$ pwd
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01
    [blyth@cms01 geant4.9.2.p01]$ cat examples/extended/persistency/gdml/README 

                            Geant4 GDML Examples
                            ====================

    This directory contains a set of examples showing the usage of the GDML
    plugin module in Geant4.

       G01        Simple example for importing and exporting simple GDML files.
       G02        Sample application showing how to import/export different
                  geometry setups, including STEP Tools files and structures
                  integrating them in a real simulation application.
       G03        Simple example showing how to import extensions to the GDML
                  schema.

    See the README file inside each example for more detail.


Module Sources
~~~~~~~~~~~~~~~

::


    [blyth@cms01 src]$ l
    total 380
    -rw-r--r--  1 blyth blyth  8601 Mar 16  2009 G4GDMLEvaluator.cc
    -rw-r--r--  1 blyth blyth  7595 Mar 16  2009 G4GDMLParameterisation.cc
    -rw-r--r--  1 blyth blyth  2380 Mar 16  2009 G4GDMLParser.cc
    -rw-r--r--  1 blyth blyth 10731 Mar 16  2009 G4GDMLRead.cc
    -rw-r--r--  1 blyth blyth 15769 Mar 16  2009 G4GDMLReadDefine.cc
    -rw-r--r--  1 blyth blyth 20242 Mar 16  2009 G4GDMLReadMaterials.cc
    -rw-r--r--  1 blyth blyth 24378 Mar 16  2009 G4GDMLReadParamvol.cc
    -rw-r--r--  1 blyth blyth  3768 Mar 16  2009 G4GDMLReadSetup.cc
    -rw-r--r--  1 blyth blyth 57935 Mar 16  2009 G4GDMLReadSolids.cc
    -rw-r--r--  1 blyth blyth 24461 Mar 16  2009 G4GDMLReadStructure.cc
    -rw-r--r--  1 blyth blyth  9860 Mar 16  2009 G4GDMLWrite.cc
    -rw-r--r--  1 blyth blyth  5343 Mar 16  2009 G4GDMLWriteDefine.cc
    -rw-r--r--  1 blyth blyth  8457 Mar 16  2009 G4GDMLWriteMaterials.cc
    -rw-r--r--  1 blyth blyth 17571 Mar 16  2009 G4GDMLWriteParamvol.cc
    -rw-r--r--  1 blyth blyth  2644 Mar 16  2009 G4GDMLWriteSetup.cc
    -rw-r--r--  1 blyth blyth 36180 Mar 16  2009 G4GDMLWriteSolids.cc
    -rw-r--r--  1 blyth blyth 13421 Mar 16  2009 G4GDMLWriteStructure.cc
    -rw-r--r--  1 blyth blyth  9239 Mar 16  2009 G4STRead.cc
    [blyth@cms01 src]$ 
    [blyth@cms01 src]$ 
    [blyth@cms01 src]$ pwd
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/persistency/gdml/src
    [blyth@cms01 src]$ 


Building GDML module in Geant4
---------------------------------

Building the GDML module in Geant4 is optional; by default, the GDML plug-in is not built 
along with the rest of the Geant4 libraries. In order to build the Geant4 module for GDML, 
one needs to have: 

* The XercesC parser pre-installed (presently GDML uses either XercesC 2.8.0 or 3.1.1 versions); 
* The following environment variables set at the time the Geant4 libraries get built:  

     * XERCESROOT, specifying the path where the XercesC parser library and headers are installed in the system;  
     * G4LIB_BUILD_GDML set to "1". 
 
Once the above setup is defined in the user's environment, the GDML module in Geant4 
will be built using the standard build procedure applicable for Geant4. 


Schema Highlights
-------------------

* http://service-spi.web.cern.ch/service-spi/app/releases/GDML/schema/gdml.xsd

VolumeType
    Represents a top of a geometrical sub-hierarchy not placed in space.
    None of its children can coincide with its boundary defined by an associated
    solid. Two different placements of the same logical volume represent two
    different geometrical hierarchies in space



* http://service-spi.web.cern.ch/service-spi/app/releases/GDML/schema/gdml_core.xsd
* http://service-spi.web.cern.ch/service-spi/app/releases/GDML/schema/gdml_solids.xsd

::

    <xs:element name="opticalsurface" substitutionGroup="SurfaceProperty">
    <xs:annotation>
    <xs:documentation>Optical surface used by Geant4 optical processes</xs:documentation>
    </xs:annotation>
    <xs:complexType>
    <xs:complexContent>
    <xs:extension base="SurfacePropertyType">
    <xs:attribute name="model" type="xs:string" default="glisur"/>
    <xs:attribute name="finish" type="xs:string" default="polished"/>
    <xs:attribute name="value" type="xs:string" default="1.0"/>
    </xs:extension>
    </xs:complexContent>
    </xs:complexType>
    </xs:element>



