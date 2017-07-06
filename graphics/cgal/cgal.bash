# === func-gen- : graphics/cgal/cgal fgp graphics/cgal/cgal.bash fgn cgal fgh graphics/cgal
cgal-src(){      echo graphics/cgal/cgal.bash ; }
cgal-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cgal-src)} ; }
cgal-vi(){       vi $(cgal-source) ; }
cgal-env(){      elocal- ; }
cgal-usage(){ cat << EOU


CGAL : collection of geometry packages under GPL/LGPL
============================================================== 


* http://doc.cgal.org/latest/Manual/packages.html


Repo
-----

* https://github.com/CGAL/cgal

CGAL Subdiv
-------------

* https://github.com/CGAL/cgal/blob/master/Subdivision_method_3/include/CGAL/Subdivision_method_3/internal/subdivision_hosts_impl_3.h

Refs
------

Halfedge Data Structures
    http://doc.cgal.org/latest/HalfedgeDS/index.html#Chapter_Halfedge_Data_Structures

3D Polyhedral Surface  (higher level use of above)
    http://doc.cgal.org/latest/Polyhedron/index.html#chapterPolyhedron

Surface Mesh (index based alternative to the above)
    http://doc.cgal.org/latest/Surface_mesh/index.html#Chapter_3D_Surface_mesh
    builds on BGL

Boost Graph Library (BGL)
    http://www.boost.org/doc/libs/1_55_0/libs/graph/doc/quick_tour.html




EOU
}
cgal-dir(){ echo $(local-base)/env/graphics/cgal ; }
cgal-cd(){  cd $(cgal-dir); }
cgal-get(){
   local dir=$(dirname $(cgal-dir)) &&  mkdir -p $dir && cd $dir

   [ ! -d cgal ] && git clone https://github.com/CGAL/cgal 

}
cgal-doc(){ open http://doc.cgal.org/latest/Manual/installation.html ; }

