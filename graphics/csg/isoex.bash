# === func-gen- : graphics/csg/isoex fgp graphics/csg/isoex.bash fgn isoex fgh graphics/csg
isoex-src(){      echo graphics/csg/isoex.bash ; }
isoex-source(){   echo ${BASH_SOURCE:-$(env-home)/$(isoex-src)} ; }
isoex-vi(){       vi $(isoex-source) ; }
isoex-env(){      elocal- ; }
isoex-usage(){ cat << EOU

IsoEx : Isosurface Extraction from volume data such as CSG (LGPL)
===================================================================

* NB Unlike OpenMesh IsoEx is under LGPL

IsoEx - Feature Sensitive Surface Extraction

The IsoEx package provides some simple classes and algorithm for isosurface
extraction. Its main purpose is to provide a sample implementation of the
Extended Marching Cubes algorithm. 

* https://www.graphics.rwth-aachen.de/software/
* https://www.graphics.rwth-aachen.de/media/resource_files/IsoEx-1.2.tar.gz

Related
---------

* https://graphics.ethz.ch/Downloads/Publications/Tutorials/2006/Bot06b/eg06-tutorial.pdf
* ~/opticks_refs/EuroGraphicsTutorial_TriangleMeshModeling_eg06-tutorial.pdf

Geometric Modeling Based on Triangle Meshes

p10:

    Notice that Marching Cubes computes intersection points on the edges of a
    regular grid only, which causes sharp edges or corners to be “chopped of”. A
    faithful reconstruction of sharp features would instead require additional
    sample points within the cells containing them. The extended Marching Cubes
    [KBSS01] therefore examines the distance function’s gradient ∇F to detect those
    cells and to find additional sample points by intersecting the tangent planes
    at the edge intersection points. This principle is depicted in Fig. 8, and a 3D
    example of the well known fandisk dataset is shown in Fig. 9. An example
    implementation of the extended Marching Cubes based on the OpenMesh data
    structure [BSM05] can be downloaded from [Bot05a].


Docs
-------

* http://www.graphics.rwth-aachen.de/IsoEx

* You have to place OpenMesh and IsoEx in the same directory for the includes to work.
* There is an example application located in IsoEx/Apps/emc.



EOU
}


isoex-url(){ echo https://www.graphics.rwth-aachen.de/media/resource_files/IsoEx-1.2.tar.gz ; }
isoex-nam(){ echo IsoEx ; }
isoex-dir(){ echo $(local-base)/env/graphics/csg/$(isoex-nam) ; }
isoex-cd(){  cd $(isoex-dir); }

isoex-get(){
   local dir=$(dirname $(isoex-dir)) &&  mkdir -p $dir && cd $dir

   local url=$(isoex-url)
   local nam=$(isoex-nam)
   local tgz=$(basename $url)

   [ ! -f "$tgz" ] && curl -L -O $url 
   [ ! -d "$nam" ] && tar zxvf $tgz 


}
