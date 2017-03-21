# === func-gen- : graphics/piston/piston fgp graphics/piston/piston.bash fgn piston fgh graphics/piston
piston-src(){      echo graphics/piston/piston.bash ; }
piston-source(){   echo ${BASH_SOURCE:-$(env-home)/$(piston-src)} ; }
piston-vi(){       vi $(piston-source) ; }
piston-env(){      elocal- ; }
piston-usage(){ cat << EOU

PISTON
=======

* http://datascience.dsscale.org
* http://datascience.dsscale.org/tools/
* http://viz.lanl.gov/projects/PISTON.html
* https://github.com/lanl/PISTON
* http://viz.lanl.gov/projects/piston.pdf
* ~/opticks_refs/LANL_MarchingCubes_Isosurface_piston.pdf


An open-source, portable cross-platform framework for data-parallel
visualization operators The Portable Data-Parallel Visualization and Analysis
Library (also referred to as PISTON) is a cross-platform software library
providing frequently used operations for scientific visualization and analysis.
The algorithms for these operations are specified in a data-parallel way. By
using nVidia’s freely downloadable Thrust library and our own tools, we can
generate executable codes for different acceleration hardware architectures
(GPUs and multi-core CPUs) from a single version of source code. The library is
designed to be extensible and is intended to be integrated into other
visualization applications.



EOU
}
piston-dir(){ echo $(local-base)/env/graphics/PISTON ; }
piston-cd(){  cd $(piston-dir); }
piston-mate(){ mate $(piston-dir) ; }
piston-get(){
   local dir=$(dirname $(piston-dir)) &&  mkdir -p $dir && cd $dir


   [ ! -d PISTON ] && git clone https://github.com/lanl/PISTON


}
