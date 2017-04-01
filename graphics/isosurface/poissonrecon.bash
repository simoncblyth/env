# === func-gen- : graphics/isosurface/poissonrecon fgp graphics/isosurface/poissonrecon.bash fgn poissonrecon fgh graphics/isosurface
poissonrecon-src(){      echo graphics/isosurface/poissonrecon.bash ; }
poissonrecon-source(){   echo ${BASH_SOURCE:-$(env-home)/$(poissonrecon-src)} ; }
poissonrecon-vi(){       vi $(poissonrecon-source) ; }
poissonrecon-env(){      elocal- ; }
poissonrecon-usage(){ cat << EOU

Poisson Recon : fitting to get an SDF from a point set 
=======================================================


Screened Poisson Surface Reconstruction 
(and Smoothed Signed Distance Reconstruction)
Version 9.01

* http://www.cs.jhu.edu/~misha/MyPapers/ToG13.pdf
* http://www.cs.jhu.edu/~misha/Code/PoissonRecon/Version9.01/
* https://github.com/mkazhdan/PoissonRecon

* http://www.cs.jhu.edu/~misha/MyPapers/ToG13.pdf
* ~/opticks_refs/Screened_Poisson_Surface_Reconstruction_Michael_Kazhdan_ToG13.pdf 


Poisson surface reconstruction creates watertight surfaces from oriented point
sets. In this work we extend the technique to explicitly incorporate the points
as interpolation constraints. The extension can be interpreted as a
generalization of the underlying mathematical framework to a screened Poisson
equation. In contrast to other image and geometry processing techniques, the
screening term is defined over a sparse set of points rather than over the full
domain. We show that these sparse constraints can nonetheless be integrated
efficiently. Because the modified linear system retains the same finite-element
discretization, the sparsity structure is unchanged, and the system can still
be solved using a multigrid approach. Moreover we present several algorithmic
improvements that together reduce the time complexity of the solver to linear
in the number of points, thereby enabling faster, higher-quality surface
reconstructions.


TK Port
------------

* http://www.vtkjournal.org/browse/publication/718
* https://github.com/daviddoria/PoissonReconstruction

This is a port of Misha Kazhdan's work (http://www.cs.jhu.edu/~misha/Code/PoissonRecon/) to VTK.
NOTE: The normals of the input polydata must be vtkFloatArray, NOT vtkDoubleArray!

PCL incorporation
--------------------

See pcl- they have integrated this into a documented library
for surface recon.



EOU
}
poissonrecon-dir(){ echo $(local-base)/env/graphics/isosurface/PoissonRecon ; }
poissonrecon-cd(){  cd $(poissonrecon-dir); }
poissonrecon-mate(){ mate $(poissonrecon-dir) ; }
poissonrecon-get(){
   local dir=$(dirname $(poissonrecon-dir)) &&  mkdir -p $dir && cd $dir

   [ ! -d PoissonRecon ] && git clone https://github.com/mkazhdan/PoissonRecon

}
