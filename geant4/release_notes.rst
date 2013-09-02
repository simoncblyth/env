Geant4 Release Notes
=====================

What changed between 4.9.2 to 4.9.5 (3 years of Geant4 development)

Highlighting changes in optical photons, geometry, visualization, persistency. 


Schedule
--------

Geant4.9 yearly update schedule::

    4.9.2 (December 19th, 2008)
    4.9.3 (December 18th, 2009)
    4.9.4 (December 17th, 2010)
    4.9.5 (December 2nd, 2011) 
    4.9.6 (November 30th, 2012


4.9.2 
------

* http://geant4.cern.ch/support/ReleaseNotes4.9.2.html 

4.9.3 
------

* http://geant4.cern.ch/support/ReleaseNotes4.9.3.html 

* GDML: it is now possible to customise the reader as well as the writer to treat ad-hoc user extensions to the GDML schema. The GDML writer now also supports dumping of optical surface properties associated with surfaces and materials.
* Measured angular reflectivity distributions for common surface treatments and reflectors have been incorporated as look-up tables (LUT).
* New default builder for optical photon physics has been added: G4OpticalPhysics together with its messenger, G4OpticalPhysicsMessenger.

* Improved ability of visualization system to render Boolean solids. Certain cases involving very near co-planar faces used to result in incorrect visualization (though not incorrect tracking). An existing mechanism to deal with this has been improved; this mechanism used to shift slightly the position of one solid in a chosen direction. Now it attempts shifts in different directions by varying small amounts until one is found that does not give an error. The system still can not handle every situation. However it results in far fewer visualization inaccuracies than were seen previously in such difficult cases.

* gMocrenFile: First Beta release of this new visualization driver to output data in the gdd format used by the gMocren volume visualization tool. gMocrenFile can include volume data from sensitive detector volumes, or from command-based scoring meshes, in addition to detector geometry and trajectories. Use new command /vis/scene/add/psHits to add scorer hits to gMocren output. For details, see the Application Developers Guide and the gMocren web site.

    * http://geant4.kek.jp/gMocren/

4.9.4 
-----

* http://geant4.cern.ch/support/ReleaseNotes4.9.4.html 

Solids (Boolean) 
Introduced recursive algorithm in CreatePolyhedron() for
Boolean operations: it uses HepPolyhedronProcessor from 'graphics_reps' module,
using new technique in attempt to avoid numerical problems for the calculation
of the polyhedron in BooleanProcessor. It allows to try all permutations, also
for Booleans of Booleans. Helps in reducing the number of cases of "Error in
Boolean processor" for visualization; still some stubborn cases are left.

   * I observed these in 4.9.2 when exporting VRML


4.9.5
------

* http://geant4.cern.ch/support/ReleaseNotes4.9.5.html 


4.9.6
------

* http://geant4.cern.ch/support/ReleaseNotes4.9.6.html

Solids (BREPs)
Added deprecation warning for BREPs module, planned for removal from next major release.

Solids (Specific)
New rewrite of G4TessellatedSolid resulting in reduced memory footprint by 50%
and large speedup; for number of facets in the thousands the speedup factor
observed is tens or more. Adopted to ad-hoc voxelization of the surface for
fast retrieval of intersected facets. Measured a factor of thousands speedup
when number of facets is of the order of order of a hundred thousand. New
classes introduced, G4SurfBits and G4SurfaceVoxelizer.  Modified
GetPointOnSurface() to throw sqrt(r) uniformly for disk surfaces, in
G4Paraboloid, G4Polycone and G4TwistedTubs.

OpenGL
Improved rendering speed through major re-design of scene and transient processing.


