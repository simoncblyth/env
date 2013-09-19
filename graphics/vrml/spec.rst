VRML97 (2.0) Specification
===========================

Intro
------

* http://en.wikipedia.org/wiki/VRML

The Web3D Consortium has been formed to further the collective development of
the format. VRML (and its successor, X3D), have been accepted as international
standards by the International Organization for Standardization (ISO).  The
first version of VRML was specified in November 1994. This version was
specified from, and very closely resembled, the API and file format of the Open
Inventor software component, originally developed by SGI. The current and
functionally complete version is VRML97 (ISO/IEC 14772-1:1997). VRML has now
been superseded by X3D (ISO/IEC 19775-1)

Spec
-----

* http://www.web3d.org/x3d/specifications/vrml/
* http://www.web3d.org/x3d/specifications/vrml/ISO-IEC-14772-VRML97/

Metadata
---------

* http://graphcomp.com/info/specs/sgi/vrml/spec/part1/nodesRef.html#Anchor
* http://doc.instantreality.org/documentation/nodetype/Anchor/

::

    187     // VRML codes are generated below
    188 
    189     fDest << "#---------- SOLID: " << pv_name << "\n";
    190 
    191     if ( IsPVPickable() ) {
    192 
    193      fDest << "Anchor {" << "\n";
    194      fDest << " description " << "\"" << pv_name << "\"" << "\n";
    195      fDest << " url \"\" " << "\n";
    196      fDest << " children [" << "\n";
    197     }
    198 
    199     fDest << "\t"; fDest << "Shape {" << "\n";
    200 
    201     SendMaterialNode();
    202 
    203     fDest << "\t\t" << "geometry IndexedFaceSet {" << "\n";



