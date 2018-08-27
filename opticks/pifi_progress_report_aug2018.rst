PIFI Progress Report August 2018
===================================

Priorities : putting ease of use first 
-----------------------------------------

Prior to starting my PIFI work at IHEP in May 
I received several requests from users for simple 
examples demonstrating Opticks usage. Describing 
Opticks usage to people with no experience of it  
made me realise that the Opticks workflow and architecture 
was focussed on the needs of development rather than the needs of users. 
This realisation together with my goal of enabling anyone 
to apply Opticks GPU acceleration to their simulation has 
led me to focus during my first months at IHEP
on making several major changes that make it drastically 
easier to apply Opticks to any geometry.

Direct geometry translation
-----------------------------

The primary change is an entirely new direct geometry translation 
approach that converts the Geant4 geometry model directly into 
the Opticks equivalent including both fully analytic and triangulated 
forms ready for upload to the GPU with a single executable, 
with no need for intermediate exports and imports.  Formerly 
triangulated geometry from G4DAE and analytic geometry from GDML were 
handled separately using a separate processing stage implemented in python
to convert the GDML into Opticks CSG geometry format. 
With the direct approach both triangulated and analytic geometry are treated 
in an integrated way with the python conversions and CSG tree balancing 
reimplemented in C++. The direct approach has been extensively validated 
by comparisons of geometry serialisation buffers. In addition I implemented 
geometry export into the glTF 3D file format, enabling visualization in
external renderers. 
The direct approach enables users to bring their detector geometry to 
the GPU using a very simple interface to an embedded Opticks. 
As well as being drastically easier for users it can also lead 
to future reductions in the Opticks codebase with several 
sub-projects and dependencies related to import 
and export of geometry being eliminated. 

Adopt modern CMake configuration best practices
-------------------------------------------------

Opticks is comprised of ~20 subprojects organized according to their dependencies on 
external packages and on each other. Adoption of modern CMake configuration best practices
has enabled automatic handling of transitive dependencies yielding greatly simplified
Opticks configuration both internally and in external usage.  Users need only 
configure against a single direct dependency, the rest of the dependency tree 
including Geant4 being configured automatically. A further benefit is the flexibility 
with which any Opticks subproject can now be tested. 
 

Prototype user example with embedded Opticks interface 
-------------------------------------------------------

Opticks is intended to be embedded within Geant4 based applications
using an interface package focussed on doing only this.  
The interface package was developed in tandem with a prototype 
Geant4 example project which aims to be very similar to other Geant4 
optical examples with a mimimum of additional code required to provide GPU accelerated
optical photon propagation. The requirememts are to pass geometry for direct GPU translation, 
collect gensteps for propagation, perform the GPU propagation and collect hits. 
Enabling the usage of embedded Opticks to be simple, self-contained and minimalistic 
is what motivated the development of the direct geometry translation approach. 


Two executable validation structure
-------------------------------------








 

Summary
-----------

There has already been substantial progress in all five areas of planned work. 
Although several aspects such as moving to direct geometry translation and 
adoption of modern configuration practices, were not explicitly forseen in 
my work plan these enhancements greatly simplify and add new possibilites in all areas. 
By adopting a user centric perspective from the outset my work has been guided by what is 
needed to meet the urgent need to integrate GPU accelerated optical photon propagation 
with the JUNO simulation framework, bringing unprecedented performance to JUNO simulation.



