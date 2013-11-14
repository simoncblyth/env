Planning Geant4 Collada Exporter
=================================

.. contents:: :local:

About Collada
--------------

General Collada notes are in :doc:`/graphics/collada/index`
keep this area for Geant4 specifics related to Collada.

Geant4 docs 
------------

* http://geant4.cern.ch/support/userdocuments.shtml
* http://www-geant4.kek.jp/Reference/

How to implement ?
--------------------

Adding collada library dependencies likely to be problematic for G4 inclusion,
nevertheless can learn from them the style of dae to aim for and generate
with XERCESC.

* http://collada.org/mediawiki/index.php/Portal:COLLADA_DOM
* https://github.com/sbarthelemy/collada-dom
* http://pycollada.github.io/creating.html


Triangulate the output geometry ?
----------------------------------

Collada can express higher level shapes, however for
widest applicability it is probably better to triangulate at export
rather than relying on subsequent triangulation. Especially as
need to define in/out materials somehow.

Collada Double Sided Faces
--------------------------

* :google:`collada double sided faces`


Sketchup DAE exporter
~~~~~~~~~~~~~~~~~~~~~~

The DAE exporter options of Sketchup are relevant:

* http://www.sketchup.com/
* http://en.wikipedia.org/wiki/SketchUp
* http://help.sketchup.com/de/article/140409

The Export two-sided faces checkbox is used to export faces twice, once for
the front and once for the back. This option doubles the number of polygons in
the resulting DAE file and can slow down rendering. However, this option
ensures that your model will appear more like it appears in SketchUp. Both
faces will always render, and materials applied to front and back faces are
preserved. When this option is selected, SketchUp will weld the vertices of the
front faces together and the vertices of the back faces together.

XML format
--------------

Borrow XML xercesc handling approach from GDML exporter.

* http://xerces.apache.org/xerces-c/program-dom-2.html
* http://xerces.apache.org/xerces-c/apiDocs-3/classDOMImplementationRegistry.html


CADMesh : From CAD files into Geant4 G4TessellatedSolid
----------------------------------------------------------

Opposite direction to what I am interested in, but illuminating nevertheless.

* https://github.com/christopherpoole/CADMesh
* http://christopherpoole.github.io/A-CAD-interface-for-GEANT4/
* http://christopherpoole.github.io/static/pdfs/Poole%20et%20al.%20-%20A%20CAD%20interface%20for%20GEANT4.pdf

The GEANT4 (version GEANT4.9.5 patch 1) geometry hierarchy is divided into 
solids, logical volumes and physical volumes where solids describe shape, logical volumes 
define material properties and mother-daughter relations, and physical volumes define 
placement within the mother volume [7, 8]. The equivalent 
GEANT4 solid to a BREP geometry is the tessellated solid (G4TessellatedSolid ) 
and has specific properties so as to enable correct geometry navigation by the 
GEANT4 kernel. 
In particular, the tessellated solid must describe a closed surface, 
that is to say the boundary between the inside and outside of the solid is defined 
for all points. Furthermore, all faces on this surface (whether they be triangular 
or quadrangular faces), must not have coincident vertexes; for example a triangular 
face must have exactly three unique vertexes thereby ensuring it has an 
area greater than zero. When a face is added to the solid, the direction and order 
of the vertexes must be anti-clockwise when the normal of the face is pointing towards 
the inside of the volume; the GEANT4 navigator uses this convention when 
it determines if a point is inside or outside of the tessellated solid. For tessellated 
solids with face vertex order in the opposite direction, navigation errors may arise. 
Additionally, boolean operations can not be performed with tessellated solids. 

As common BREP CAD file formats and GEANT4 tessellated solids encode 
a surface mesh in similar ways, a direct mapping can be achieved by iterating 
over all of the faces in a BREP and adding them directly to the tessellated solid 
in GEANT4. For a direct mapping to be effective however, vertexes and faces 
defined in a BREP must be accessible in a common way that is independent 
of the source file format. This common access to various CAD file formats is 
achieved here using VCGLIB. Various custom types derived from VCGLIB base 
classes may be defined so as to describe for example a stereo-lithography format 
(STL) [1] or Stanford polygon file format (PLY) [6] mesh and the elements that 
encode it, such as vertexes, edges and faces. Template parameters are used to...




