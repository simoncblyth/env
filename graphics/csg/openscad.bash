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
