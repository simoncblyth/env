Other Geant4 Geometry Approaches
================================

STEP
-----

* http://geant4.web.cern.ch/geant4/UserDocumentation/UsersGuides/ForApplicationDeveloper/html/ch04.html

The Geant4 solid modeler is STEP compliant. STEP is the ISO standard defining
the protocol for exchanging geometrical data between CAD systems. This is
achieved by standardizing the representation of solid models via the EXPRESS
object definition language, which is part of the STEP ISO standard.

The STEP standard supports multiple solid representations. Constructive Solid
Geometry (CSG) representations and Boundary Represented Solids (BREPs) are
available. Different representations are suitable for different purposes,
applications, required complexity, and levels of detail. CSG representations
are easy to use and normally give superior performance, but they cannot
reproduce complex solids such as those used in CAD systems. BREP
representations can handle more extended topologies and reproduce the most
complex solids.

All constructed solids can stream out their contents via appropriate methods and streaming operators.


Geometry Source
----------------

::

    [blyth@belle7 DDDB]$ pwd
    /data1/env/local/dyb/NuWa-trunk/dybgaudi/Detector/XmlDetDesc/DDDB


HepRep
-------

* java based 


Geant4 VMC
------------

* http://root.cern.ch/drupal/content/geant4-vmc
* http://root.cern.ch/drupal/content/vmc
* http://indico.cern.ch/getFile.py/access?contribId=51&sessionId=7&resId=0&materialId=slides&confId=66646

Geant4 VMC represents the realisation of the Virtual Monte Carlo (VMC) for
Geant4 . It can be also seen as a Geant4 application implemented via the VMC
interfaces. It implements all Geant4 user mandatory classes and user action
classes, which provide the default Geant4 VMC behaviour, that can be then
customized by a user in many ways.



InSTEP
-------

* http://www.solveering.com/instep.htm

InStep supports import & export of the following formats:

#. STL (Ascii/Text and Binary)
#. OBJ
#. DAE (for triangulated bodies)
#. VRML
#. X3D
#. GDML (with limited definition of domains/materials)


