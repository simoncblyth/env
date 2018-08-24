# === func-gen- : presentation/presentation fgp presentation/presentation.bash fgn presentation fgh presentation
presentation-src(){      echo presentation/presentation.bash ; }
presentation-source(){   echo ${BASH_SOURCE:-$(env-home)/$(presentation-src)} ; }
presentation-vi(){       vi $(presentation-source) ; }
presentation-env(){      elocal- ; }
presentation-usage(){ cat << EOU

Presentation preparation
============================


IDEAS : JUNO Collab Meeting Plenary July 2018
----------------------------------------------

Objective : drum up interest sufficient to get some profs to assign some workers

* interest in Opticks, expts 
* Opticks roadmap


Presentations online
----------------------

* http://simoncblyth.bitbucket.io

Shrinking PDFs with ColorSync Utility
---------------------------------------

* https://www.cultofmac.com/481247/shrink-pdfs-colorsync-utility/

* use "Live update from filter inspector"
* copied the existing "Reduce File Size" filter
* modified "Reduce File Size Copy" removed Image Sampling, set Image Compression to JPEG 75% 
  
  * PDF size reduced from 49M to 15M and quality reduction difficult to notice

::

    epsilon:Desktop blyth$ du -hs opticks*
     49M	opticks_gpu_optical_photon_simulation_jul2018_chep.pdf
     15M	opticks_gpu_optical_photon_simulation_jul2018_chep_j75.pdf


Steps
~~~~~~~

1. spotlight search for "ColorSync Utility.app"
2. select "Filters" tab
3. find the "Reduce File Size Copy" filter, open the 
   disclosure to see the JPEG compression factor that will be applied
4. use "File > Open" in "ColorSync Utility" to open the uncompressed PDF
5. pick the "Reduce File Size Copy" using drop down menu at bottom left
6. press the "Apply" button at bottom right
7. use "File > Save As" to save the result to a different name

Compare sizes::

    epsilon:Desktop blyth$ du -hs opticks_gpu_optical_photon_simulation_jul2018_chep*
     15M	opticks_gpu_optical_photon_simulation_jul2018_chep.pdf
     49M	opticks_gpu_optical_photon_simulation_jul2018_chep_uncompressed.pdf 


TODO
-----

* DONE : enable issues on bitbucket
* prepare a list of email addresses of the Opticks interested
* write Opticks News

  * whats new : CMake 3.5+, BCM, target export -> easy config 
  * WIP : direct from G4 geometry
  * issues enabled on bitbucket
  * mailing list : ground rules, not an issue tracker : use bitbucket for that

* send Opticks News to the interested list, invite to mailing list, inform on issue tracker

* https://groups.io/g/opticks/promote

  * embed signup form into bitbucket 


Machinery Fixes
----------------

* Makefile had some old simoncblyth.bitbucket.org rather than simoncblyth.bitbucket.io
* apache changes in High Sierra, see hapache-


CHEP 2018, Sofia, Bulgaria
----------------------------

* review developments since last CHEP,  
 
  * notes/progress.rst 
  * okc-vi has some month-by-month history

Timeline
------------

::

    Mon June 25  * start preparing talk, submit a version of presentation before July 02 
    Mon July 02  
    Mon July 09  * 1st day of CHEP


::

    Dear simon blyth,

    The CHEP 2018 Program Committee and Track Conveners are happy to announce
    that the parallel and poster sessions are published in Indico at
    https://indico.cern.ch/event/587955/timetable/#all.detailed
    Please check your assigned slot and contact us if:

    1. There is an error in the speaker name or incorrect speaker is assigned;
    2. The speaker is "For the <xxx> collaboration", we need the real speaker name;
    3. You have found another error.

    Note that you can upload your presentation/poster at any time, but please do so
    at the latest one week before the start of the conference, i.e. by 2 July 2018.
    Else your contribution risks not being taken into account for the summary of its track.

    We take this opportunity to remind you that:

    - the parallel presentation time format is 15 min - 12' presentation + 3' Q/A.
    - the posters should be in A0 format and will be displayed for the duration
    of the conference.

    Thank you for your help and see you soon in Sofia!

                                         Best Regards, the Program Committee.



Potential Customers
-----------------------

::

    SNO+     : large scale liquid scintillator expt 
    WATCHMAN : 
    THEIA
    CheSS   


Sep 2017 Wollongong 
---------------------

22nd Geant4 Collaboration Meeting, UOW Campus, Wollongong (Australia), 25-29 September 2017.


intro
~~~~~~

The main detector consists of a 35.4 m (116 ft) diameter transparent acrylic
glass sphere containing 20,000 tonnes of linear alkylbenzene liquid
scintillator, surrounded by a stainless steel truss supporting approximately
53,000 photomultiplier tubes (17,000 large 20-inch (51 cm) diameter tubes, and
36,000 3-inch (7.6 cm) tubes filling in the gaps between them), immersed in a
water pool instrumented with 2000 additional photomultiplier tubes as a muon
veto.[8]:9 Deploying this 700 m (2,300 ft) underground will detect neutrinos
with excellent energy resolution.[3] The overburden includes 270 m of granite
mountain, which will reduce cosmic muon background.[9]

The much larger distance to the reactors (compared to less than 2 km for the
Daya Bay far detector) makes the experiment better able to distinguish neutrino
oscillations, but requires a much larger, and better-shielded, detector to
detect a sufficient number of reactor neutrinos.



renders to make
~~~~~~~~~~~~~~~~~~

* j1707 InstLODCull 
* j1707 analytic



Dear Visualisation, Geometry and EM Coordinators,..

I think my ongoing work on accelerating optical photon simulation
using the NVIDIA OptiX GPU ray tracer in the context of PMT based
neutrino detectors such as JUNO or Daya Bay may be of interest 
to your subgroups. 

I presented my work at the 2014 Okinawa Collaboration Meeting.
My work has matured greatly since then, if I were to receive an
invitation to attend the upcoming Geant4 Collaboration Meeting in Australia
I may be able to secure funding to attend, enabling me to present/discuss
my work with everyone interested in GPU accelerated optical photon 
simulation and GPU visualization. 

My work is available in the open source Opticks project

* https://bitbucket.org/simoncblyth/opticks/

Numerous status reports, conference presentations and videos 
are linked from https://simoncblyth.bitbucket.io including the 
latest focusing on CSG on GPU.

* https://simoncblyth.bitbucket.io/env/presentation/opticks_gpu_optical_photon_simulation_jul2017_ihep.html
 (it takes a several seconds for the slide presentation to load)

Over the past 6 months I have succeeded to implement general 
CSG binary tree intersection on the GPU within an NVIDIA OptiX 
intersect "shader" and have used this capability to develop 
automatic translation of GDML geometries into a fully 
analytic GPU appropriate forms, using the glTF 3D file format 
as an intermediate format.
Entire detector geometries including all material and optical surface properties
are translated, serialized into buffers and copied to the GPU 
at initialization. 

This allows full analytic geometries, without triangulation approximation, 
to be auto-translated and copied to the GPU meaning that in principal 
GPU ray intersections should very closely match those obtained with Geant4, 
as the GPU and CPU are doing the same thing, finding roots of  
polynomials with the same coefficients.  OpenGL/OptiX GPU instancing 
techniques are used to efficiently handle geometries with ~30k PMTs and 
BVH (boundary volume heirarchy) structure acceleration comes
for free with the NVIDIA OptiX ray tracing framework.

GPU CSG geometry together with CUDA/OptiX ports of optical photon 
physics allows optical photons to be simulated entirely on the GPU. 
Photon generation (from G4Cerenkov, G4Scintillation) 
and propagation (G4OpAbsorption, G4OpBoundaryProcess, G4OpRayleigh)
have been ported.
Of the many millions of optical photons simulated per event
only the small fraction that hit photon detectors need to be copied 
to the CPU where they can populate the standard Geant4 hit collections.
Integration with Geant4 is currently done via modified G4Scintillation
and G4Cerenkov processes which collect "gensteps" that are copied to the GPU.

Simulations limited by optical photons can benefit hugely from 
the GPU parallelism that Opticks makes accessible with optical
photon speedup factors estimated to exceed 1000x for the JUNO 
experiment.

Opticks needs to mature through production usage within the JUNO
experiment(and perhaps others) before it makes sense to consider details
of any "formal" integration, nevertheless looking ahead
I am interested to learn opinions of Geant4 members as to whether
that direction is even feasible ? And what form it might take if it were.

The bottom line is that I think a significant number of experiments
can benefit greatly from using Opticks together with Geant4 and I would
like to help them do so.

Simon C. Blyth,  National Taiwan University


update CHEP talk : ie intro to someone never having seen Opticks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* review progress since CHEP (~1 year) doing appropriate updates, analytic CSG  



Dear Laurent, 

Thanks.   I’ve recently made some progress that I guess will be particularly interesting 
to you and the Visualisation group,  on improving OpenGL performance for very large geometries,
specifically the JUNO Neutrino detector: 

   18,000 20inch PMTs of ~5000 triangles each, 
    36,000 3inch PMTs of ~1500 triangles each,
   nominal total of ~150M triangles

Using dynamic instance culling and variable level-of-detail meshes for instances ( the PMTs) 
based on distance to the instance.  These use GPU compute 
(OpenGL transform feedback streams with geometry shader, so not NVIDIA specific) 
prior to rendering each frame in order to skip instances that are not visible and replace 
distant instances with simpler geometry.   

On my Macbook Pro 2013 laptop with NVIDIA GT 750M the optimisation turns a formerly painful ~2fps
experience into  >30fps  of comfortable interactivity. Plus the performance can be tuned for the available
GPU by adjusting the distances at which to switch on different levels of detail.  

Its a rather neat technique as the optimisation is cleanly split from the actual rendering,
which remains unchanged other that getting its 4x4 instance transforms from the dynamic buffers
for each level of detail.

I’ll hope we’ll find you a time slot to present your work.

Me too!  How long a presentation would you like ?


Simon





Jul 2017 IHEP : Page by page
-------------------------------

Intro p1
~~~~~~~~~

* moving optical photon generation/propagation to GPU gives drastic speedups
* also offloads memory requirements to GPU, only hits need to come to CPU 
* will not repeat introduction, there are a few context slides at back 
* will focus on a new feature of Opticks ...

"Outline" p2 : cf LLR
~~~~~~~~~~~~~~~~~~~~~~~~~

* starting from summary at LLR last November, 
  there I described chi2 comparisons establishing that 
  the GPU ported optical physics matches Geant4 

  * however there was a large caveat : for match need analytic geometry on GPU
  
* to overcome the geometry limitation : I have developed a general GPU CSG intersection algorithm
* moving CSG to the GPU makes it possible to automate conversion of GDML files into a GPU optimized form 
    

CSG p3  
~~~~~~~~~

* primitive shapes combined using set operations : union, intersection and difference
* how to find intersections between rays and geometry .... ?

  * use parametric definition of ray -> parametric-t identifies position along the ray   
  * for primitives described implicitly, just find roots of t to get intersections
  * composite intersects will be the intersect with one of the primitives 

    * ... just need to pick the correct intersect according to the CSG expression

CSG Roth, High Perf GPU p4
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* traditional way is to find intervals for all primitives and 
  recursively combine them according to the CSG expression

  * not well suited to GPU, high perf on GPU requires...

CSG Single Hit p5
~~~~~~~~~~~~~~~~~~~~

* alternate single hit approach, treats intersects with a pair of primitives 
  as a simple state machine : avoids store/sort intervals

  * hits classified using the normal at intersection, combined into an 
    action using lookup tables for the CSG operator and for which is closer

  * loop action handles intersects that need to be disqualified

* natural way to implement would be to use recursion, 
  BUT: OptiX does not support recursion within intersect programs


CSG Tree Serialization p6
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* in order to get the CSG tree to the GPU need to serialize it...
* have adopted a complete binary tree serialization : as this simplifies 
  tree handling on the GPU 

* regular pattern of bits in complete binary tree, enables navigation just 
  by bit shifts... 

* postorder tree traverse starts from leftmost and always visits children before
  their parent (same order that postorder recursion would yield)

  * bit twiddling can reproduce the postorder traverse

CSG Intersect Pseudocode p7
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* iterative algorithm ... arranged via 
  nested while loops, the outer over tranches and the inner over
  nodes within the tranches, allows recursion to be emulated (also recursion within recursion...)

* recursive "return" is emulated via pushing onto a stack of intersects
* end result is picking the "winning" intersect    


CSG Primitives p8
~~~~~~~~~~~~~~~~~~~~

* flexibility/expressiveness of CSG means that you do not need many primitives,
  only around 10 needed 

  * some requirements : must be solid (ie closed) 


CSG Primitives Parade p9
~~~~~~~~~~~~~~~~~~~~~~~~~~


CSG Primitives Parade Code p10
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* python code defining CSG trees, serialized to file in OpticksCSG format 


CSG Primitives : what is included p11
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* on GPU : CUDA functions for bounding box and ray geometry intersection
* on CPU : SDFs 


CSG balancing deep trees p12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* CSG implementation intended for individual solids, not scenes
* CSG difference inconvenient, as not commutative, hence convert to +ve form
  allowing 


Dayabay ESR p13
~~~~~~~~~~~~~~~~


CSG_CYLINDER subtraction -> speckle in hole p14
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Coincident Faces p15
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SDFs p16
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


SDF Isosurface Extraction p17
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Translated Solid Debugging p18
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Full Analytic GDML Workflow p19
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Screenshots p20
~~~~~~~~~~~~~~~~~~~


Summary p25
~~~~~~~~~~~~~~





Workflow
-----------

Preparation workflow:

#. change presention name below and create the .txt
#. iterate on presentation by running the below which invokes *presentation-make*
   and *presentation-open* to view local html pages in Safari::

   presentation.sh

#. static .png etc.. are managed within bitbucket static repo, 
   local clone at ~/simoncblyth.bitbucket.org

   * HMM NOW ~/simoncblyth.bitbucket.io 

   * remember not too big, there is 1GB total repo limit 

#. running presentation.sh updates the derived html within
   the static repo clone, will need to "hg add" to begin with


Publishing to remote:

#. update index page as instructed in *bitbucketstatic-vi*
#. push the statics to remote 


#. creating PDFs, see slides-;slides-vi add eg "slides-get-jul2017" 
   depending on slide count



Creating retina screencaptures
---------------------------------

* shift-cmd-4 and drag out a marquee, this writes .png file to Desktop

::

   cd ~/env/presentation   ## cd to appropriate directory for the capture

   osx_                    ## precursor define the functions
   osx_ss-cp name          

   ## copy last screencapture from Desktop to corresponding relative dir beneath ~/simoncblyth.bitbucket.org 
   ## this is the local clone of the bitbucket statics repo


Incorporating retina screencaptures
-------------------------------------

::

    simon:presentation blyth$ cd ~/simoncblyth.bitbucket.org/
    simon:simoncblyth.bitbucket.org blyth$ downsize.py env/graphics/ggeoview/PmtInBox-approach.png
    INFO:env.doc.downsize:Resize 2  
    INFO:env.doc.downsize:downsize env/graphics/ggeoview/PmtInBox-approach.png to create env/graphics/ggeoview/PmtInBox-approach_half.png 2138px_1538px -> 1069px_769px 
    simon:simoncblyth.bitbucket.org blyth$ 



s5 rst underpinning
--------------------

* http://docutils.sourceforge.net/docs/user/slide-shows.html#s5-theme-files


Presentation Structure
------------------------

* https://hbr.org/2012/10/structure-your-presentation-li

* What-Is/What-could-be 


Opticks Overview : Narrative Arc
-----------------------------------


* optical photon problem of neutrino detectors

* NVIDIA OptiX

Let me pick up the story from ~1yr ago Oct 2016 (CHEP) 

* approx 1yr ago, Opticks was mostly using tesselated approximate geometries
  with some manual PMT analytic conversions

* after lots of work : validation chi-sq comparisons were showing an excellent 
  match between the GPU ported simulation and standard G4,
  BUT with a great big CAVEAT : **purely analytic geometry**

  * so i had got the optical physics to the GPU  
  * BUT: only approximate tesselated geometry, with some manual analytic  


Requirements were clear:

* intersection code for rays with CSG trees 
* auto-conversion of GDML into OpticksCSG for upload to GPU 


Additions
------------

* Large Geometry Techniques : instance culling, LOD
* Primitives : Torus, Ellipsoid
* CSG non-primitives, complement 

 

presentations review, highlighting new developments
------------------------------------------------------

2017
~~~~~


opticks_gpu_optical_photon_simulation_sep2017_wollongong.txt
     update oct2016_chep with 1 year of progress (mainly analytic CSG)

opticks_gpu_optical_photon_simulation_jul2017_ihep.txt
     mostly on CSG

opticks_gpu_optical_photon_simulation_jan2017_psroc.txt
     same as LLR 

2016
~~~~~~

opticks_gpu_optical_photon_simulation_nov2016_llr.txt
    focus on optical photon validation comparisons in simple analytic geomtry

    * tconcentric : message "match achieved after many fixes" 
    * excellent match, loath to loose that with approximate geometry -> analytic implementation


opticks_gpu_optical_photon_simulation_oct2016_chep.txt
    validation start, chisq minimization
    
    * ioproc-open


opticks_gpu_optical_photon_simulation_jul2016_weihai.txt

opticks_gpu_optical_photon_simulation_may2016_lecospa.txt

opticks_gpu_optical_photon_simulation_april2016_gtc.txt

opticks_gpu_optical_photon_simulation_march2016.txt

optical_photon_simulation_progress.txt
     Opticks : GPU Optical Photon Simulation using NVIDIA OptiX
     Jan 2016

opticks_gpu_optical_photon_simulation_psroc.txt
     Jan 2016 

opticks_gpu_optical_photon_simulation.txt
     Jan 2016 



2015
~~~~~
    
optical_photon_simulation_with_nvidia_optix.txt
     Optical Photon Simulation with NVIDIA OptiX
     July 2015

gpu_optical_photon_simulation.txt
     200x Faster Optical Photon Propagation with NuWa + Chroma ?
     Jan 2015

gpu_accelerated_geant4_simulation.txt
     GPU Accelerated Geant4 Simulation with G4DAE and Chroma
     Jan 2015 
 
2014
~~~~~

g4dae_geometry_exporter.txt
     G4DAE : Export Geant4 Geometry to COLLADA/DAE XML files
     19th Geant4 Collaboration Meeting, Okinawa, Sept 2014


g4dae_geometry_exporter (Sept 2014)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**G4DAE : Export Geant4 Geometry to COLLADA/DAE XML files**

9th Geant4 Collaboration Meeting, Okinawa, Sept 2014

Why export into DAE ?

Ubiquitous geometry visualization for Geant4 users and outreach. 
Facilitate innovative use of geometry data.

Moving geometry to GPU and implementing simple shaders
unleases performant visualization 

* Exporter details
* What is COLLADA/DAE ?
* Validating exports : compare with GDML and VRML2
* General viewing of exports
* Custom use : bridging to GPU
* OpenGL Viewer implementation
* Optical Photon Data handling
* Introducing Chroma
* Chroma raycasting
* Chroma photon propagation
* G4DAE exporter status


gpu_accelerated_geant4_simulation (Jan 2015)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**GPU Accelerated Geant4 Simulation with G4DAE and Chroma** 


* Geometry Model Implications
* Geant4 <-> Chroma integration : External photon simulation workflow
* G4DAE Geometry Exporter
* Validating GPU Geometry
* G4DAEChroma bridge
* Chroma forked
* Validating Chroma generated photons
* Next Steps
* Visualizations 

Track > Geometry intersection typically limits simulation performance. 
Geometry model determines techniques (and hardware) available to accelerate
intersection.

*Geant4 geometry model (solid base)*
    Tree of nested solids composed of materials, each shape represented by different C++ class

*Chroma Geometry model (surface based)*
    List of oriented triangles, each representing boundary between inside and outside materials.

3D industry focusses on surface models >> frameworks 
and GPU hardware designed to work with surface based geometries.


*Geometry Set Free*

Liberating geometry data from Geant4/ROOT gives free choice of visualization
packages. Many commercial, open source apps/libs provide high performance
visualization of DAE files using GPU efficient OpenGL techniques. 
Shockingly Smooth Visualization performance

**Above not really true: better to say surface based geometry is a better fit for ray tracing**

BUT Chroma needs : triangles + inside/outside materials

Chroma tracks photons through a triangle-mesh detector geometry, simulating
processes like diffuse and specular reflections, refraction, Rayleigh
scattering and absorption. Using triangle meshes eliminate geometry code as
just one code path.

Optical photons (modulo reemission) are the leaves of the simulation tree,
allowing external simulation to be integrated rather simply.

* generation of Cerenkov and Scintillation Photons based on Geant4 Generation Step inputs


gpu_optical_photon_simulation (Jan 2015)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**200x Faster Optical Photon Propagation with NuWa + Chroma ?**

Fivefold path:

* Export G4 Geometry as DAE
* Convert DAE to Chroma
* Chroma Validation, raycasting
* Chroma Stability/Efficiency Improvements Made
* G4/Chroma Integration
* Chroma vs G4 Validation 


2015-01-06 : last commit to Chroma fork
2015-01-20 : first commit to Opticks "try out NVIDIA Optix 301" https://bitbucket.org/simoncblyth/opticks/commits/bd1c43

 
optical_photon_simulation_with_nvidia_optix (July 2015}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Optical Photon Simulation with NVIDIA OptiX**

*OptiX : performance scales with CUDA cores across multiple GPUs*

* Why not Chroma ?
* Introducing NVIDIA OptiX

  * Parallels between Realistic Image Synthesis and Optical Simulation
  * OptiX Programming Model

* OptiX testing

  * OptiX raycast performance
  * OptiX Performance Scaling with GPU cores

* New Packages Replacing Chroma

  * Porting Optical Physics from Geant4/Chroma into OptiX
  * Optical Physics Implementation
  * Random Number Generation in OptiX programs (initialization stack workaround)
  * Fast material/surface property lookup from boundary texture
  * Reemission wavelength lookup from Inverted CDF texture
  * Recording the steps of ~3 million photons 
  * Scintillation Photons colored by material
  * Indexing photon flag/material sequences
  * Selection by flag sequence

* Mobile GPU Timings
* Operation with JUNO Geometry ?
* Next Steps


opticks_gpu_optical_photon_simulation_psroc (Jan 2016)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Opticks : GPU Optical Photon Simulation**

**Opticks integrates Geant4 simulations with state-of-the-art NVIDIA OptiX GPU ray tracing**

* DayaBay, JUNO can expect: Opticks > 1000x G4 (workstation GPUs)


* Neutrino Detection via Optical Photons
* Optical Photon Simulation Problem
* NVIDIA OptiX GPU Ray Tracing Framework
* Brief History of GPU Optical Photon Simulation Development
* Introducing Opticks : Recreating G4 Context on GPU

* Validating Opticks against Theory

  * Opticks Absolute Reflection compared to Fresnel expectation
  * Opticks Prism Deviation vs Incident angles for 10 wavelengths
  * Multiple Rainbows from a Single Drop of Water

* Validating Opticks against Geant4

  * Disc beam 1M Photons incident on Water Sphere (S-Pol)
  * 2ns later, Several Bows Apparent
  * Rainbow deviation angles
  * Rainbow Spectrum for 1st six bows
  * 1M Rainbow S-Polarized, Comparison Opticks/Geant4

* Opticks Overview

  * Performance Comparison Opticks/Geant4

Three levels of geometry:

* OptiX: analytic intersection code
* OpenGL: tesselation for visualization
* Geant4: geometry construction code in CfG4 package


opticks_gpu_optical_photon_simulation (Jan 2016, Xiamen)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Opticks : GPU Optical Photon Simulation**

Executive Summary

* Validation comparisons with Geant4 started, rainbow geometry validated
* Performance factors so large they become irrelevant (>> 200x)

* Brief History of GPU Optical Photon Simulation Development
* Introducing Opticks : Recreating G4 Context on GPU
* Large Geometry/Event Techniques
* Mesh Fixing with OpenMesh surgery

  * G4Polyhedron Tesselation Bug
  * OpenMeshRap finds/fixes cleaved meshes

* Analytic PMT geometry description

  * Analytic PMT geometry : more realistic, faster, less memory
  * Analytic PMT in 12 parts instead of 2928 triangles
  * OptiX Ray Traced Analytic PMT geometry
  * Analytic PMTs together with triangulated geometry
 
* Opticks/Geant4 : Dynamic Test Geometry

  * Opticks Absolute Reflection compared to Fresnel expectation

* Rainbow Geometry Testing

  * Opticks/Geant4 Photon Step Sequence Comparison


Optical Photons now fully GPU resident
All photon operations now done on GPU:

* seeded (assigned gensteps)
* generated
* propagated
* indexed material/interaction histories



opticks_gpu_optical_photon_simulation_march2016
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Opticks : GPU Optical Photon Simulation**


* Validation comparisons with Geant4 advancing, single PMT geometry validated


* PmtInBox test geometry
* Single PMT Geometry Testing
* PmtInBox at 1.8ns
* PMT Opticks/Geant4 Step Sequence Comparison

  * Good agreement reached, after several fixes: geometry, TIR, GROUPVEL
  * nearly identical geometries (no triangulation error)

  * PMT Opticks/Geant4 step comparison TO BT [SD] : position(xyz), time(t)
  * PMT Opticks/Geant4 step comparison TO BT [SD] : polarization(abc), radius(r)
  * PmtInBox Opticks/Geant4 Chi2/ndf distribution comparisons
  * PmtInBox issues : velocity of photon propagation

* Photon Propagation Times Geant4 cf Opticks
* External Photon Simulation Workflow


opticks_gpu_optical_photon_simulation_april2016_gtc
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Opticks : Optical Photon Simulation for Particle Physics with NVIDIA® OptiX™**

Nothing much new in this one.


opticks_gpu_optical_photon_simulation_may2016_lecospa
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Nothing much new in this one. Lots of intro.


opticks_gpu_optical_photon_simulation_oct2016_chep (20 pages, ~15 min)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


opticks_gpu_optical_photon_simulation_nov2016_llr  (32 pages, ~30 min)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Opticks Event Buffers  (technicality)
* Time/Memory profile of multi-event production mode running (~technicality)

* tconcentric : test geometry configured via boundaries
* tconcentric : fixed polarization "Torch" light source
* tconcentric : spherical GdLS/LS/MineralOil
* tconcentric : Opticks/Geant4 chi2 comparison
* tconcentric : Opticks/Geant4 history counts chi2/df ~ 1.0

* Group Velocity problems -> Time shifts  (~technicality)





EOU
}
presentation-dir(){ echo $(env-home)/presentation ; }
presentation-c(){   cd $(presentation-dir); }
presentation-cd(){  cd $(presentation-dir); }

presentation-ls(){   presentation-cd ; ls -1t *.txt ; }
presentation-txts(){ presentation-cd ; vi $(presentation-ls) ;  }


#presentation-name(){ echo gpu_accelerated_geant4_simulation ; }
#presentation-name(){ echo optical_photon_simulation_with_nvidia_optix ; }
#presentation-name(){ echo optical_photon_simulation_progress ; }
#presentation-name(){ echo opticks_gpu_optical_photon_simulation ; }
#presentation-name(){ echo opticks_gpu_optical_photon_simulation_psroc ; }

#presentation-name(){ echo opticks_gpu_optical_photon_simulation_march2016 ; }
#presentation-name(){ echo opticks_gpu_optical_photon_simulation_april2016_gtc ; }
#presentation-name(){ echo opticks_gpu_optical_photon_simulation_may2016_lecospa ; }
#presentation-name(){ echo opticks_gpu_optical_photon_simulation_jul2016_weihai ; }
#presentation-name(){ echo jnu_cmake_ctest ; }
#presentation-name(){ echo opticks_gpu_optical_photon_simulation_oct2016_chep ; }
#presentation-name(){ echo opticks_gpu_optical_photon_simulation_nov2016_llr ; }
#presentation-name(){ echo opticks_gpu_optical_photon_simulation_jan2017_psroc ; }
#presentation-name(){ echo opticks_gpu_optical_photon_simulation_jul2017_ihep ; }
#presentation-name(){ echo opticks_gpu_optical_photon_simulation_sep2017_jinan ; }
#presentation-name(){ echo opticks_gpu_optical_photon_simulation_sep2017_wollongong ; }
#presentation-name(){ echo opticks_gpu_optical_photon_simulation_jul2018_chep ; }
#presentation-name(){ echo opticks_gpu_optical_photon_simulation_jul2018_ihep ; }
presentation-name(){ echo dybdb_experience ; }


presentation-info(){ cat << EOI

    presentation-name        : $(presentation-name)
    presentation-path        : $(presentation-path)
    presentation-url-remote  : $(presentation-url-remote)
    presentation-url-local   : $(presentation-url-local)
    presentation-dir         : $(presentation-dir)


EOI
}


presentation-path(){ echo $(presentation-dir)/$(presentation-name).txt ; }
presentation-export(){
   export PRESENTATION_NAME=$(presentation-name)
}
presentation-e(){ vi $(presentation-path) ; }
presentation-edit(){ vi $(presentation-path) ; }
presentation-ed(){ vi $(presentation-path) ~/workflow/admin/reps/ntu-report-may-2017.rst ; }

presentation-make(){
   presentation-cd
   presentation-export
   env | grep PRESENTATION
   make $*
}


presentation-writeup(){
   presentation-cd
   vi opticks_writeup.rst
}

presentation-remote(){
   #echo simoncblyth.bitbucket.org
   echo simoncblyth.bitbucket.io
}

presentation-url-local(){ echo http://localhost/env/presentation/$(presentation-name).html?page=${1:-0} ; }
presentation-open(){
   open $(presentation-url-local $*)
   sleep 0.3
   slides-
   slides-safari    ## just resizes browser
} 


presentation-url-remote(){   echo http://$(presentation-remote)/env/presentation/$(presentation-name).html?page=${1:-0} ; }
presentation-open-remote(){  open $(presentation-url-remote $*) ; }

presentation--(){
   presentation-
   presentation-info
   presentation-make
   presentation-open
}




