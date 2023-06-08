chep-env(){ echo -n ; }
chep-e(){  chep-cd ; vi chep.txt ; }
chep-vi(){ vi $BASH_SOURCE ; }
chep-cd(){ cd $(env-home)/presentation/chep ; }

chep-notes(){ cat << EON
CHEP 2023 : May 8-12 
==========================

* https://www.jlab.org/conference/CHEP2023
* https://simoncblyth.bitbucket.io/


Proceedings
--------------

* https://indico.jlab.org/event/459/page/356-proceedings

CHEP 2023 will be publishing a peer-reviewed proceedings following the
conference. All presenters will be invited to contribute. Paper length is
flexible but, following prior guidelines, something on the order of 8--12 pages
(including figures) is average.

Deadlines for that stage will be finalized soon.  Presenters can anticipate a
nominal July 2023 timeframe for submission.

Additional details will be posted here when they are available.  Deadlines will
be publicized shortly.

* https://indico.jlab.org/event/459/attachments/9030/13097/epj-woc-latex.zip



Ideas for slides
-------------------

* Hidden benefits of bringing optical simulation to GPU 

  * it improves CPU simulation : due to the testing 
  * U4Recorder every point 


Abstract
----------

Opticks : GPU Optical Photon Simulation using NVIDIA OptiX 7 and NVIDIA CUDA

Opticks is an open source project that accelerates optical photon simulation by
integrating NVIDIA GPU ray tracing, accessed via the NVIDIA OptiX 7 API, with
Geant4 toolkit based simulations. A single NVIDIA Turing architecture GPU has
been measured to provide optical photon simulation speedup factors exceeding
1500 times single threaded Geant4 with a full JUNO analytic GPU geometry
automatically translated from the Geant4 geometry. Optical physics processes of
scattering, absorption, scintillator reemission and boundary processes are
implemented in CUDA based on Geant4.  Wavelength-dependent material and surface
properties as well as inverse cumulative distribution functions for reemission
are interleaved into GPU textures providing fast interpolated property lookup
or wavelength generation.

In this work we describe the near complete re-implementation of geometry and
optical simulation required to adopt the entirely new NVIDIA OptiX 7 API, with
the implementation now directly CUDA based with OptiX usage restricted to
providing intersects.  The new Opticks features a modular many small header
design that provides fine grained testing both on GPU and CPU as well as
substantial code reductions from CPU/GPU sharing.  Enhanced modularity has
enabled CSG tree generalization to support "list-nodes", similar to
G4MultiUnion, that improve performance for complex CSG solids.  Recent addition
of support for interference effects in boundaries with multiple thin layers,
such as anti-reflection coatings and photocathodes, using CUDA compatible
transfer matrix method (TMM) calculations of reflectance, transmittance and
absorptance is also reported.


PRIMARY TOPICS, AFTER CONTEXT DESCRIPTION
-------------------------------------------

15min talk (including questions) so need to aim to limit to 15 slides
(except very fast ones like renders) : but as its not me presenting 
need to make the slides rather complete and self explaining

Emphasis split between the three primary topics 60:20:20 ? 
If context takes ~5 slides, only ~10 slides : so split is 6, 2, 2 


0. context (~5 slides)

1. full re-impl (~6 slides) : Bulk of work, but software design is not so gripping to present  

  * TODO: cherry pick from presentations over past 2 years

2. CSG list nodes (~2 slides) 

   * tree balancing problem
   * more closely matching intersection approach to the details of the geometry

3. TMM  (~2 slides)



THOUGHTS

* smaller topics can use existing presentation slides almost asis 
* how to present the full re-impl not so easy 



Review vCHEP 2021 talk : what stage was I at 
-----------------------------------------------

* https://simoncblyth.bitbucket.io/env/presentation/opticks_vchep_2021_may19.html
* that talk was real short (15min), better to look at other talks at close to that time to get a more detailed view 
* CSGFoundry was brand new in separate repo, was just at geometry stage, not simulation 


Review Opticks developments since May 2021, update progress notes, decide on the topics to highlight
------------------------------------------------------------------------------------------------------

* re-implemented optical photon simulation in CUDA (not OptiX), NVIDIA OptiX 7 just used for geometry intersection
* CSG List Nodes
* TMM 


Other topics:

YES:

* "simtrace" 2D slicing  : needed as illustrations use it 

NO: Skipping topics that are not easy to explain or not fully baked

* Geant4 Z-cutting trees : ?
* Cerenkov : s2 integration
* Cerenkov : inverse transform wavelength does better in float precision that rejection sampling 
* CSG sub-sub-"bug"




Cherry pick
--------------

https://simoncblyth.bitbucket.io/env/presentation/juno_opticks_cerenkov_20210902.html 

* p17: GPU counterpart header pattern 

https://simoncblyth.bitbucket.io/env/presentation/opticks_autumn_20211019.html

* p29 : early GPU simtrace description



Need way to express that NVIDIA is proceeding with ray tracing performance inprovement
----------------------------------------------------------------------------------------


* https://www.cgdirector.com/nvidia-graphics-cards-order-performance/
* https://hothardware.com/news/nvidia-gpu-technology-delivers-a-ray-tracing-performance-lift
* https://research.nvidia.com/publication/2022-01_GPU-Subwarp-Interleaving


* turing, ampere, ada

* https://www.nvidia.com/en-gb/geforce/ada-lovelace-architecture/

  * Third-Gen RT Cores


* https://www.techpowerup.com/299092/nvidia-adas-4th-gen-tensor-core-3rd-gen-rt-core-and-latest-cuda-core-at-a-glance

Sep 21st, 2022 08:47 Discuss (22 Comments)
Yesterday, NVIDIA launched its GeForce RTX 40-series, 

The third-generation RT Core being introduced with Ada offers twice the
ray-triangle intersection performance over the "Ampere" RT core, and introduces
two new hardware components—Opacity Micromap (OMM) Engine, and Displaced
Micro-Mesh (DMM) Engine. OMM accelerates alpha textures often used for elements
such as foliage, particles, and fences; while the DMM accelerates BVH build
times by a stunning 10X. DLSS 3 will be exclusive to Ada as it relies on the
4th Gen Tensor cores, and the Optical Flow Accelerator component on Ada GPUs,
to deliver on the promise of drawing new frames purely using AI, without
involving the main graphics rendering pipeline.



* https://www.titancomputers.com/What-Is-the-Difference-Between-RTX-Core-Generations-s/1225.htm

* 20-series  1st gen RT cores  Turing 
* 30-series  2nd gen RT cores  Ampere
* 40-series  3rd gen RT cores  Ada Lovelace

* https://www.nvidia.com/en-gb/design-visualization/rtx-6000/
* https://www.nvidia.com/en-gb/technologies/ada-architecture/


* https://www.nvidia.com/en-gb/data-center/l40/

NVIDIA L40 GPU Unprecedented visual computing performance for the data center.  

* https://www.youtube.com/watch?v=PWcNlRI00jo

  GTC Sept 2022 Keynote
  
  07:42 Shader Execution Reordering 
  17:18 Turing, Ampere, Ada
  17:46 Geforce RTX 4090 

  https://images.nvidia.com/nvimages/gtc/pdf/NVIDIA_GTC2022_Fall_Highlights.pdf 

  https://images.anandtech.com/doci/17583/18947343.jpg  SER

  https://images.nvidia.com/aem-dam/Solutions/geforce/ada/ada-lovelace-architecture/nvidia-ada-gpu-science.pdf

  This is all about DLSS 3

  https://www.anandtech.com/show/17583/the-nvidia-geforce-project-beyond-and-gtc-fall-2022-keynote-live-blog-starts-at-8am-pt1500-utc 

  https://images.anandtech.com/doci/17583/19206531.jpg

  https://images.nvidia.com/aem-dam/en-zz/Solutions/technologies/NVIDIA-ADA-GPU-PROVIZ-Architecture-Whitepaper_1.0.pdf




EON
}



chep-wc(){ 
   chep-cd
   wc -w chep*_abstract.txt
}



