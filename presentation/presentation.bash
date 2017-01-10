# === func-gen- : presentation/presentation fgp presentation/presentation.bash fgn presentation fgh presentation
presentation-src(){      echo presentation/presentation.bash ; }
presentation-source(){   echo ${BASH_SOURCE:-$(env-home)/$(presentation-src)} ; }
presentation-vi(){       vi $(presentation-source) ; }
presentation-env(){      elocal- ; }
presentation-usage(){ cat << EOU

Presentation preparation
============================

Preparation workflow:

#. change presention name below and create the .txt
#. iterate on presentation by running the below which invokes *presentation-make*
   and *presentation-open* to view local html pages in Safari::

   presentation.sh

#. static .png etc.. are managed within bitbucket static repo, 
   local clone at ~/simoncblyth.bitbucket.org

   * remember not too big, there is 1GB total repo limit 

#. running presentation.sh updates the derived html within
   the static repo clone, will need to "hg add" to begin with


Publishing to remote:

#. update index page as instructed in *bitbucketstatic-vi*
#. push the statics to remote 


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

presentations 
---------------

summary
~~~~~~~~~

g4dae_geometry_exporter.txt
     G4DAE : Export Geant4 Geometry to COLLADA/DAE XML files
     19th Geant4 Collaboration Meeting, Okinawa, Sept 2014

gpu_optical_photon_simulation.txt
     200x Faster Optical Photon Propagation with NuWa + Chroma ?
     Jan 2015

gpu_accelerated_geant4_simulation.txt
     GPU Accelerated Geant4 Simulation with G4DAE and Chroma
     Jan 2015 
     
optical_photon_simulation_with_nvidia_optix.txt
     Optical Photon Simulation with NVIDIA OptiX
     July 2015

optical_photon_simulation_progress.txt
     Opticks : GPU Optical Photon Simulation using NVIDIA OptiX
     Jan 2016


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


















EOU
}
presentation-dir(){ echo $(env-home)/presentation ; }
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
presentation-name(){ echo opticks_gpu_optical_photon_simulation_nov2016_llr ; }

presentation-path(){ echo $(presentation-dir)/$(presentation-name).txt ; }
presentation-export(){
   export PRESENTATION_NAME=$(presentation-name)
}
presentation-edit(){ vi $(presentation-path) ; }
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
   echo simoncblyth.bitbucket.org
}

presentation-open(){
   open http://localhost/env/presentation/$(presentation-name).html?page=${1:-0}
   sleep 0.3
   slides-
   slides-safari 
} 

presentation-open-remote(){
   open http://$(presentation-remote)/env/presentation/$(presentation-name).html?page=${1:-0}
}

