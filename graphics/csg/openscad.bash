# === func-gen- : graphics/csg/openscad fgp graphics/csg/openscad.bash fgn openscad fgh graphics/csg
openscad-src(){      echo graphics/csg/openscad.bash ; }
openscad-source(){   echo ${BASH_SOURCE:-$(env-home)/$(openscad-src)} ; }
openscad-vi(){       vi $(openscad-source) ; }
openscad-env(){      elocal- ; }
openscad-usage(){ cat << EOU

OpenSCAD (GPLv2)
===================

* http://www.openscad.org
* http://www.openscad.org/documentation.html

OpenSCAD is a software for creating solid 3D CAD objects.
The Programmers Solid 3D CAD Modeller.

OpenSCAD builds on top of a number of free software libraries; is uses Qt for
user interface, CGAL for CSG evaluation, OpenCSG and OpenGL for CSG previews,
as well as boost, eigen and glew.





OpenSCAD2
------------

* https://github.com/doug-moen/openscad2/blob/master/rfc/Overview.md

OpenSCAD2 is a backward compatible redesign of the OpenSCAD language. The goals are:

1. to make OpenSCAD easier to use;
2. to make OpenSCAD more expressive and powerful, 
   not by adding complexity and piling on features, 
   but by making the core language simpler and more uniform, 
   and by removing restrictions on how language elements 
   can be composed together to create larger components.


**Thoughts**

Domain specific languages, seem a waste of effort to me... 
Easier to design some easy to use python classes and 
thus avoid getting people to learn the new language.



Red Mercury : code name for an experimental prototype additions to OpenSCAD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://github.com/doug-moen/openscad2/blob/master/new_geometry/Introduction.md

Red Mercury contains the following technologies:

1. It is based on Functional Representation (F-Rep), not meshes (B-Rep).
2. **It supports fast GPU-based preview, by executing the model (including CSG operations) on the GPU. 
   It does this by compiling the model into GLSL language**.
3. It supports multiple colours and materials.
4. The modeling language is based on OpenSCAD2.

RHg will contain a from-scratch implementation of the modeling language and
geometry engine. The requirements of F-Rep, and compilation to GLSL, put
significant design pressures on the modeling language. The language will be
based on OpenSCAD2, with whatever modifications turn out to be required by the
technology.


**Thoughts:**

Interesting idea, but appears to be just a proposal, no meat, very few commits. 



CSG File Format
------------------

* https://github.com/openscad/openscad/wiki/CSG-File-Format

CSG file format is a lightweight, text-based, language-independent CSG data
interchange format. It was derived from .scad format. CSG format defines a
small set of formatting rules for the portable representation of structured CSG
data.






EOU
}
openscad-dir(){ echo $(local-base)/env/graphics/csg/graphics/csg-openscad ; }
openscad-cd(){  cd $(openscad-dir); }
openscad-mate(){ mate $(openscad-dir) ; }
openscad-get(){
   local dir=$(dirname $(openscad-dir)) &&  mkdir -p $dir && cd $dir

}
