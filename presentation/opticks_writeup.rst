Opticks Writeup Preparations
===============================

CHEP Proceedings
------------------

* Deadline Feb 6th 2017
* 8 pages limit, create in ioproc style with::

    ioproc-;ioproc--

*presentation-writeup* 
    edit this document, adding text preparation notes

*workflow-;reps-;reps-edit* 
    last NTU report

References
------------

* ~/opticks_refs for references and inspiration docs
* https://cloud.github.com/downloads/thrust/thrust/Thrust%3A%20A%20Productivity-Oriented%20Library%20for%20CUDA.pdf

Figures 
--------

* how many ? what size ? 
* plots, screen shots

Tables
-------

* performance comparison 

HEP Software Paper Tone Examples
------------------------------------

* http://geant4.cern.ch/results/index.shtml
* http://geant4.cern.ch/results/publications.shtml#journal


What makes OptiX fast ?
--------------------------

SBVH : Spatial Splits in Bounding Volume Hierarchies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://dl.acm.org/citation.cfm?id=1572771
* http://www.nvidia.com/docs/IO/77714/sbvh.pdf

* http://highperformancegraphics.org/previous/www_2009/presentations/stich-spatial.pdf
* http://highperformancegraphics.org/previous/www_2009/program.html


OptiX talks
~~~~~~~~~~~~~~

* http://bps11.idav.ucdavis.edu/talks/12-userDefinedRayTracingPipelines-Parker-BPS2011.pdf


Scheduling in OptiX
~~~~~~~~~~~~~~~~~~~~~

* http://highperformancegraphics.org/previous/www_2009/presentations/nvidia-rt.pdf
* ~/opticks_refs/scheduling_in_optix_robison_nvidia-rt.pdf 

* http://highperformancegraphics.org/previous/www_2009/presentations/aila-understanding.pdf
* ~/opticks_refs/aila-understanding.pdf





Megakernels considered harmful
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://dl.acm.org/citation.cfm?id=2492060



Related Work
--------------

Purpose of *Related Work* section
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* set the scene
* fair coverage of Chroma limitations without getting down to performance comparison numbers. 
  There really is no comparison performance wise, so no point in doing the comparison work.

Chroma
~~~~~~~

* ~/opticks_refs/chroma.pdf

Chroma implements both GPU ray tracing and optical photon simulation using CUDA kernels
that are launched via PyCUDA scripts. The scripts make extensive use of NumPy. 


Chroma : why not to use
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* requires optical photons to be copied from CPU to GPU, Opticks generates photons on GPU
* implements its own GPU ray tracing, including BVH acceleration structure creation and traversal.  
  These remain active areas of computer science research with frequent algorithmic improvements.
* implements its own simple GPU ray tracing work scheduling, again an area of active research 
  which is liable to require reimplementation for new GPU architectures
* limited to triangle geometries
 
Efficient GPU ray tracing through use of acceleration structures and GPU work scheduling 
remain active areas of research with implementations liable to require reworking for each 
GPU architecture.

The Chroma approach of implementing GPU ray tracing makes it 
The ambitious nature of the Chroma implementation makes it difficult to maintain for 
production usage, but at the same time an excellent learning resource. 

Chroma is currently implemented using CUDA but in principal it could be ported to OpenCL 
to enable usage on AMD and Intel GPUs in addition to the NVIDIA GPUs that CUDA requires.

Opticks aim is to very closely reproduce Geant4 simulation results.


Ambitiously:

* constructs BVH acceleration structure using Morton codes (integer with spatial grouping)


Tao Voxelization
~~~~~~~~~~~~~~~~~~~

* http://juno.ihep.ac.cn/cgi-bin/Dev_DocDB/ShowDocument?docid=1293

  Fast Muon Simulation in the JUNO Central Detector

* http://iopscience.iop.org/article/10.1088/1674-1137/40/8/086201/pdf


Muon Photon yield
~~~~~~~~~~~~~~~~~~~~

Muon average track length in CD LS is about 23m, suppose 2MeV/cm energy deposit, LS optical photon yield is 10000/MeV
23*100*2*10000 = 46,000,000 optical photons.


* http://juno.ihep.ac.cn/Dev_DocDB/0020/002058/003/Tutorial_for_muon_reconstruction.pdf


GeantV 
~~~~~~~~~~~~~~~~~~~~


VecGeom/USolids
~~~~~~~~~~~~~~~~~

* https://arxiv.org/pdf/1312.0816v1.pdf Vectorising the detector geometry to optimize particle transport

Shared objective with Opticks so intro will cover similar ground, but for neutrino detectors
where the simplicity means GPU focus and drastic speedups are in the cards.

* :google:`vecgeom`

* https://indico.fnal.gov/getFile.py/access?contribId=8&resId=0&materialId=slides&confId=9213  (Jan 30, 2015 )

* https://indico.fnal.gov/getFile.py/access?contribId=24&sessionId=35&resId=0&materialId=slides&confId=9717

  Integration of VecGeom into Geant4 (Sept 2015)

  VECGEOM_REPLACES_USOLIDS macro 

* all navigation is still performed and controlled by Geant4
* boolean operations with solids still managed by Geant4

Hmm the above imply no migration of geometry to reside on GPU ?

* http://aidasoft.web.cern.ch/USolids
* https://indico.cern.ch/event/468478/contributions/2195688/attachments/1292009/1924788/Geom_solids_WP3_June2016.pptx
* ~/opticks_refs/Geom_solids_WP3_June2016.pdf 

* http://indico.cern.ch/event/382133/contributions/1809012/attachments/761594/1044801/geom-solids_WP3_June_2015.pdf


**Towards a high performance geometry library for particle-detector simulations**

* http://iopscience.iop.org/article/10.1088/1742-6596/608/1/012023/pdf
* ~/opticks_refs/Towards_a_high_perf_geom_lib_Apostolakis_2015_J._Phys.3A_Conf._Ser._608_012023.pdf 


VecGeom/GPU impressions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Interesting there is no mention of acceleration structures...  

Probably this is a no-triangle effect, when have millions of triangles 
the use of acceleration structures is mandatory, but operating at volume level 
makes this is much less critical. Nevertheless geometries can get to 100k of volumes
so it would still be beneficial.

* GPUs currently an "also have.." bullet point, not yet shifted mindset to "GPU resident" approach.


Opticks use of VecGeom, is it possible ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Not straightforwardly...


* :google:`optix rtPotentialIntersection rtReportIntersection`

* https://docs.nvidia.com/gameworks/content/gameworkslibrary/optix/optix_quickstart.htm


Scene data is provided to OptiX via a node tree structure that 
is constructed using the OptiX CPU side API. 
Geometry nodes at the leaves of the tree are assigned 
primitive counts and associated CUDA programs that return bounding box and 
report ray primitive intersection positions. 
The programs access GPU geometry buffers which are copied from the CPU at initialization.
Multiple geometry instance nodes are allowed to share geometry objects, this 
together with the use of transform nodes is used to model the many thousands of 
photomultiplier tubes without repetition of information.
The OptiX node tree is also used to configure acceleration structures
used for different parts of the scene.








OptiX intersection programs use interleaved calls rtPotentialIntersection and rt

::

￼




Overview
----------

Following the end of the free lunch of CPU performance improvement 
in approximately 2005 improvements increasingly requires development 
effort to adopt new concurrent technologies.







Most work towards accelerating Geant4 simulation performance has been 
done in the context of vectorization 

Most neutrino detectors are comprised of large homogenous target volumes such as
scintillating liquids coupled with photon detectors such as photomultiplier tubes.
Simplicity of design and 
The relative simplicity of neutrino detector 

Neutrino detectors 

The relative simplicity of neutrino detectors and the
optical physics that 

The simplicity of optical physics and neutrino detector geometry 
makes it feasible 


Most existing work on improving the performance
of simulation 


GPU or MIC
~~~~~~~~~~~~

* :google:`GPU or MIC`
* https://www.karlrupp.net/2013/06/cpu-gpu-and-mic-hardware-characteristics-over-time/

* http://www.nvidia.com/object/justthefacts.html Intel Xeon Phi vs NVIDIA GPU



Introduction
-------------

approach
~~~~~~~~~

Neutrino detector simulations are dominated by the propagation of
very large numbers of optical photons.  This shines a harsh light 
on Geant4.

importance of simulation and geant4
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* mission critical for all phases high energy physics experiments from detector design, ... 

* geant4 dominance, common element in software of most experiments

neutrino detection special features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* simple geometry compared to collider experiments with large homogenous 
  target volumes coupled with many thousands of nearly identical photon detectors

* photons created by only a few processes, mainly interested in small subset 
  of photons that hit photon detectors

* important signal and background events can yield many millions of photons

* era of ever increasing CPU is over

* HEP is lagging in its transition to parallel computing and especially to the GPU, 
  due the complexity of most detector geometries and many of the physics processes, however
  neutrino detectors are special 

* simplicity of optical physics, the independence of each photon and the sheer number
  of optical photons makes the simulation of optical photons to be well suited 
  to use of GPU massive parallelism techniques

* unification of graphics and computation



NVIDIA OptiX
--------------

* http://on-demand-gtc.gputechconf.com/gtcnew/on-demand-gtc.php?searchByKeyword=OptiX&searchItems=&sessionTopic=&sessionEvent=&sessionYear=&sessionFormat=&submit=&select=


GPU Ray Tracing Exposed: Under the Hood of the NVIDIA OptiX Ray Tracing Engine 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

http://on-demand.gputechconf.com/gtc/2010/video/S12250-GPU-Ray-Tracing-NVIDIA-Optix-Ray-Tracing-Engine.mp4
 
* 1st 15 min discusses Aila and Lane paper 
* last 15 min discusses the state machine 

* key observation : divergence is temporary, usually comes back to common states


Aila
~~~~~

* https://mediatech.aalto.fi/~timo/


Steven Parker, Stanford University Lecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

https://www.youtube.com/watch?v=KH3hS0LBq80

* jump to 23min to skip the general intro

* SIMT divergence kept temporary via prescribed state transition ordering
* termination penalty
* many threads, means must minimize state per thread

GPU radiation therapy
~~~~~~~~~~~~~~~~~~~~~~

* https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4003902/pdf/nihms564930.pdf



OpenLab on HEP usage of parallelism 
-------------------------------------

* :google:`Jarp S, Lazzaro A and Nowak A`

* http://inspirehep.net/record/1211475/files/jpconf12_396_042043.pdf?version=1
* Many-core experience with HEP software at CERN openlab


GPU within HEP
----------------

* :google:`GPU use within High Energy Physics`


* https://arxiv.org/pdf/1512.06637.pdf

  On the Way to Future’s High Energy Particle Physics Transport Code  (For the GeantV collab)

* has a simple description of AtRest, AlongStep, PostStep, ...


impressions
~~~~~~~~~~~~

Other than Chroma have so far found no mention of 
acceleration structures (eg BVH) in discussions of GPU ports
of G4. This suggests that 


:google:`geant4 accelerate`
--------------------------------

GPU in Physics Computation: Case Geant4 Navigation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

2011 investigation of G4 GPU porting of Geant4 navigation.

* https://arxiv.org/pdf/1209.5235v1.pdf
* ~/opticks_refs/1209.5235v1.pdf 

**Impression**

* **no mention of acceleration structures, suggests stuck in G4 geometry straightjacket/mindset** 
* direct ports of highly branching CPU code to GPU yields little benefit
* porting is not enough, re-architecting is needed  


High energy electromagnetic particle transportation on the GPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://iopscience.iop.org/article/10.1088/1742-6596/513/5/052013/pdf



Vectorising the detector geometry to optimize particle transport (Dec 2013)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://arxiv.org/abs/1312.0816


Free Lunch is Over
-------------------

* http://www.drdobbs.com/web-development/a-fundamental-turn-toward-concurrency-in/184405990
* http://dl.acm.org/citation.cfm?id=1095421
* ~/opticks_refs/p54-sutter.pdf

* http://www.gotw.ca/publications/concurrency-ddj.htm



:google:`intel Many Integrated Core vs NVIDIA GPU`
-----------------------------------------------------

* http://www.greymatter.com/corporate/hardcopy-article/gpu-vs-manycore/  (2012)


Experience with Intel's Many Integrated Core architecture in ATLAS software
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

http://iopscience.iop.org/article/10.1088/1742-6596/513/5/052018/pdf

Intel’s MIC architecture is very new, and perhaps put in the market somewhat
prematurely, if the state of the tools, drivers, and other support are an
indicator. Rapid progress is expected to continue, but at this point it is pure
speculation which tools Intel will endure long-term.  The next generation of
MIC boards, Knight’s Landing, will be available both as a coprocessor and as
standalone CPU. It is not clear whether the latter will allow memory
extensions, but Intel has stated that the standalone product will remove the
current programming complexities of data transfers, suggesting a single main
memory. Other indicated improvements are in memory bandwidth and energy
efficiency, but no word on an increase in the number of cores.  Our
object-oriented codes scattered across shared libraries run more effectively on
the MIC than on Xeon CPUs if there is sufficient, fully independent,
parallelism. Thus, if a future board allows memory extensions, we may well be
able to run process-parallel AthenaMP or event- parallel GaudiHive efficiently
on the MIC. Until then, work should focus on the service model that allows
effective use of accelerator cards.


GPGPU for track finding in High Energy Physics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://arxiv.org/abs/1507.03074


Proceedings, GPU Computing in High-Energy Physics (GPUHEP2014) : Pisa, Italy, September 10-12, 2014
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://inspirehep.net/record/1387408?ln=en

* http://www-library.desy.de/preparch/desy/proc/proc14-05/40.pdf  some contribs seems truncated


:google:`Photon Propagation with GPUs in IceCube`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://bib-pubdb1.desy.de/record/291395
* http://adsabs.harvard.edu/abs/2013NIMPA.725..141C


