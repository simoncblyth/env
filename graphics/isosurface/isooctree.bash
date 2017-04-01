# === func-gen- : graphics/isosurface/isooctree fgp graphics/isosurface/isooctree.bash fgn isooctree fgh graphics/isosurface
isooctree-src(){      echo graphics/isosurface/isooctree.bash ; }
isooctree-source(){   echo ${BASH_SOURCE:-$(env-home)/$(isooctree-src)} ; }
isooctree-vi(){       vi $(isooctree-source) ; }
isooctree-env(){      elocal- ; }
isooctree-usage(){ cat << EOU

Unconstrained Isosurface Extraction on Arbitrary Octrees
==========================================================

* http://www.cs.jhu.edu/~misha/Code/IsoOctree/
* http://www.cs.jhu.edu/~misha/MyPapers/SGP07a.pdf
* ~/opticks_refs/Unconstrained_IsoSurface_Extraction_Arbitary_Octree_SGP07a.pdf

The code provides an implementation of the isosurface extraction in the context
of mesh-simplification. Given an input mesh, the Euclidean Distance Transform
of the mesh is computed and an octree is adapted to the data, sampling the
transform more finely near the surface of the mesh. (Additional parameters
allow the user to restrict octree refinement to high-curvature regions.)
Finally, the zero-crossing isosurface is computed and written to the specified
output file. (By default, the output is a polygonal representation, though a
triangulation can be obtained as well by enabling the flag that computes the
minimal area triangulation.) Note: The construction of the EDT assumes that the
input mesh is watertight with properly oriented triangles. In the case that
these conditions are not satisfied (and sometimes even if they are) spurious
disconnected surface patches may be generated.


* http://www.cs.jhu.edu/~misha/Code/
* https://github.com/mkazhdan/IsoSurfaceExtraction

* https://github.com/mkazhdan/PoissonRecon/tree/master/Src


Observations
--------------

* many compiler errors ... 
* old, from 2007 
* only a ZIP, not in a repo 
* perhaps, other projects have move evolved versions, see poissonrecon-


VTK Port
------------

* http://www.vtkjournal.org/browse/publication/718
* https://github.com/daviddoria/PoissonReconstruction

This is a port of Misha Kazhdan's work (http://www.cs.jhu.edu/~misha/Code/PoissonRecon/) to VTK.
NOTE: The normals of the input polydata must be vtkFloatArray, NOT vtkDoubleArray!



Source Look
---------------

Uncertain that can cleanly split the phases:

* mesh -> octTree  (need to replace this with SDF->octTree )
* octTree -> mesh 

::

 413 template<class NodeData,class Real,class VertexData>
 414 template<class Vertex>
 415 int IsoOctree<NodeData,Real,VertexData>::set(const std::vector<Vertex>& vertices,const std::vector<std::vector<int> >& polygons,
 416                                              const int& maxDepth,const int& setCenter,const Real& flatness,
 417                                              Point3D<Real>& translate,Real& scale,const int& noTransform)
 418 {
 419     this->maxDepth=maxDepth;
 420     OctNode<NodeData,Real>::NodeIndex nIdx;
 421 
 422     MeshInfo<double> mInfo;
 423     std::vector<int> myTriangles;
 424     mInfo.set(vertices,polygons,Real(1.1),translate,scale,noTransform);
 425     myTriangles.resize(mInfo.triangles.size());
 426     for(int i=0;i<int(mInfo.triangles.size());i++) myTriangles[i]=i;
 427 
 428     cornerValues.clear();
 429     Real dist;
 430     Point3D<Real> n,p;
 431     for(int c=0;c<Cube::CORNERS;c++)
 432     {
 433         int x,y,z;
 434         Cube::FactorCornerIndex(c,x,y,z);
 435         p[0]=Real(x);
 436         p[1]=Real(y);
 437         p[2]=Real(z);
 438 
 439         setDistanceAndNormal(myTriangles,mInfo,p,dist,n);
 440         cornerValues[OctNode<NodeData,Real>::CornerIndex(nIdx,c,maxDepth)]=VertexData(dist,n);
 441     }
 442     if(setCenter)
 443     {
 444         Real w;
 445         OctNode<NodeData,Real>::CenterAndWidth(nIdx,tree.nodeData.center,w);
 446         setDistanceAndNormal(myTriangles,mInfo,tree.nodeData.center,dist,n);
 447         tree.nodeData.v=VertexData(dist,n);
 448     }
 449     setChildren(&tree,nIdx,myTriangles,mInfo,maxDepth,setCenter,flatness);
 450     return 1;
 451 }




EOU
}
isooctree-dir(){ echo $(local-base)/env/graphics/isosurface/IsoOctree ; }
isooctree-cd(){  cd $(isooctree-dir)/$1 ; }
isooctree-mate(){ mate $(isooctree-dir) ; }
isooctree-get(){
   local dir=$(dirname $(isooctree-dir)) &&  mkdir -p $dir && cd $dir

   local url=http://www.cs.jhu.edu/~misha/Code/IsoOctree/IsoOctree.zip
   local dst=$(basename $url)

   [ ! -f "$dst" ] && curl -L -O $url  
   [ ! -d "IsoOctree" ] && unzip -d IsoOctree $dst    # its an exploding zip

}


isooctree-patch()
{
   isooctree-cd SRC
   perl -pi -e 's,stdext::hash_map,std::unordered_map,g' *
   perl -pi -e 's,<hash_map>,<unordered_map>,g' * 
}

isooctree-make()
{

   clang Main.cpp -std=c++11 -I. -Wno-c++11-compat-deprecated-writable-strings -Wno-format -Wno-tautological-compare

}


