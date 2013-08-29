Geant4 CAD interface
======================

Usually the direction is from CAD files into Geant4.
I want to go the other way, to allow using Geant4 geometries on GPUs.


Importing CAD models as tesselated solids
-------------------------------------------

* http://geant4.web.cern.ch/geant4/UserDocumentation/UsersGuides/ForApplicationDeveloper/html/ch04.html

Tessellated solids can also be used to import geometrical models from CAD
systems (see Figure 4.1). In order to do this, it is required to convert first
the CAD shapes into tessellated surfaces. A way to do this is to save the
shapes in the geometrical model as STEP files and convert them to tessellated
(faceted surfaces) solids, using a tool which allows such conversion. At the
time of writing, at least two tools are available for such purpose: STViewer
(part of the STEP-Tools development suite) or FASTRAD. This strategy allows to
import any shape with some degree of approximation; the converted CAD models
can then be imported through GDML (Geometry Description Markup Language) into
Geant4 and be represented as G4TessellatedSolid shapes.

Other tools which can be used to generate meshes to be then imported in Geant4 as tessellated solids are:

* STL2GDML - A free STL to GDML conversion tool.

* SALOME - Open-source software allowing to import STEP/BREP/IGES/STEP/ACIS formats, mesh them and export to STL.

* ESABASE2 - Space environment analysis CAD, basic modules free for academic non-commercial use. Can import STEP files and export to GDML shapes or complete geometries.

* CADMesh - Tool based on the VCG Library to read STL files and import in Geant4.

* Cogenda - Commercial TCAD software for generation of 3D meshes through the module Gds2Mesh and final export to GDML.


STEP export from Geant4 ?
---------------------------

* :google:`geant4 export STEP EXPRESS`

* http://deslab.mit.edu/DesignLab/dicpm/step.html  STEP and EXPRESS


G4tgb
------

::

    [blyth@cms01 ascii]$ vi include/G4tgbGeometryDumper.hh
    [blyth@cms01 ascii]$ pwd
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/persistency/ascii
    [blyth@cms01 ascii]$ 

   
::

   G4tgbVolumeMgr* volmgr = G4tgbVolumeMgr::GetInstance()
   volmgr->AddTextFile(theFileName) 
 
::

   G4tgbGeometryDumper::GetInstance()->DumpGeometry(theFilename) 




FreeCAD
--------

* http://cad-gdml.in2p3.fr/
* https://edms.in2p3.fr/file/I-026238/2/CAD-GDML_17-06-2011.pdf


* http://csrsrv1.fynu.ucl.ac.be/csr_web/geant/step-gdml.php
* http://minimalcms.sourceforge.net/demo/proxy/apps/phpbb/free-cad/viewforum.php?f=10&sid=2cf1eca707ab5e7e8b6fad536af0c6dc
* http://en.wikipedia.org/wiki/FreeCAD


ESA uses restrictive licences
-----------------------------------

* http://cao-gdml.in2p3.fr/presentations/GSantin_CAD_Geant4_ESA_v10.pdf

CAD interfaces for simulation and analysis tools in the space domain, 
Giovanni Santin
Geant4 CAD interface development meeting 
Orsay, 23 March 2010 


CADMesh
--------

* http://code.google.com/p/cadmesh/

Importing predefined CAD models into GEANT4 is not always possible or requires
intermediate file format conversion to Geometry Description Markup Language
(GDML) using commercial or third party software. CADMesh is a direct CAD model
import interface for GEANT4 optionally leveraging VCGLIB, and ASSIMP by
default. Currently it supports the import of triangular facet surface meshes
defined in formats such as STL and PLY. A G4TessellatedSolid is returned and
can be included in a standard user detector constructor.

Additional functionality is included for the fast navigation of tessellated
solids by automatically creating equivalent tetrahedral meshes thereby making
smart voxelisation available for the solid.

Based on VCGLIB 

Meshlab
--------

* http://meshlab.sourceforge.net/  

Can import/export: VRML, X3D, COLLADA, STL, PLY
Based on VCGLIB


VCGLIB 
-------

http://vcg.isti.cnr.it/~cignoni/newvcglib/html/


salome
-------

* http://www.salome-platform.org/

#. Create/modify, import/export (IGES, STEP, BREP), repair/clean CAD models
#. Mesh CAD models, edit mesh, check mesh quality, import/export mesh (MED, UNV, DAT, STL)
#. Handle physical properties and quantities attached to geometrical items



