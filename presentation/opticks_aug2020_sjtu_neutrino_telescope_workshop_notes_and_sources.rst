opticks_aug2020_sjtu_neutrino_telescope_workshop_notes_and_sources
=====================================================================

What to cover
--------------

* Opticks aims to provide a full simulation equivalent to Geant4 (and validated against it)

* processes that are implemented : Absorption, Rayleigh Scattering

  * URL reference to the rayleigh scattering  

* processes that are not implemented : Mie Scattering 

* straight forward to add your own parameterized scattering (simple CUDA function)

* CG has been photon mapping for 30 years 

* Provide a way to validate optimization tricks 


Title
-------

::

   Opticks: GPU photon simulation via NVIDIA OptiX ;  
   Applied to neutrino telescope simulations ?  


Performance Studies for the KM3NeT Neutrino Telescope
--------------------------------------------------------

* https://antares.in2p3.fr/Publications/thesis/2010/Claudio-Kopper-phd.pdf
* ~/opticks_refs/Performance_Studies_for_the_KM3Net_Neutrino_Telescope_Claudio-Kopper-phd.pdf


p49

As simulating each single photon would take far too much CPU time to be
feasible for Monte Carlo mass productions, an alternative approach using pre-
simulated photon tables is used: Photons are only simulated once using the
Geant3 [76] based helper tool gen, which records the light output of a short
muon segment (with a length of about 1 to 2m) or of an electromagnetic shower
on concentric spheres. Several of these spheres are placed around the emitters
at different distances. All photons intersecting the spheres are recorded. A
second helper tool, hit, subsequently divides these spheres into angular bins
and converts the photon fields into tables containing hit prob- abilities.
These probabilities include the OMs’ characteristics such as their angular
acceptance and their wavelength-dependent quantum efficiency. Thus, the photon
tables produced by hit contain the full set of OM properties and have to be
re-calculated for each type of OM that is to be simulated.


geasim, a full tracking Monte Carlo code based on Geant3 [76]. This tool
provides the same functionality as km3, with the exception of light scattering.
It does not use a table-based approach, but rather simulates each particle
using the full Geant3 engine. To be reasonably fast, light is only propagated
in straight lines, which enables the code to speed up processing by a fair
amount, as most photons can simply be rejected by simple geometric
considerations: a photon with a straight path that can never hit an OM can be
rejected early in the code. This is of course only true if light scattering is
neglected. As geasim is still quite slow when simulating very long muon tracks,
it is mainly used for the hadronic part of neutrino interactions near the
vertex. It can be combined with km3 by using a special mode where only the
hadronic component is simulated by geasim, whereas the muon is simulated by
km3.


p59

Once a muon has reached the detector, it has to be propagated through its
sensitive volume. At the same time, secondary particles from the muon need to
be tracked and all Cherenkov light needs to be propagated through the detector
medium. A simulation code based on Geant4 [91] was developed for this purpose.
The code performs a full tracking simulation of every particle inside the
sensitive volume. This includes the simulation of every single Cherenkov
photon. This approach takes an immense amount of computing time, but it
provides accurate event simulations usable for cross-checks of parameterised
simulations.


p63

Most of these simulated photons, however, will never reach an optical module,
as a neutrino detector is only sparsely instrumented. In a way, most of the
processing time is wasted on the simulation of photons that will never be seen.
An easy solution to this problem can be found if optical scattering of photons
is neglected. In this case, the simulation code can decide if a photon will
never reach an optical module by purely geometric calculations before even
starting the tracking code. Whole segments of a muon track can be skipped in
the simulation, as unscattered light from these segments would never reach any
optical module. Note that, in a Monte Carlo tracking simulation, this is only
possible when light scattering is neglected, because the code cannot know if a
photon that would not reach an OM on its direct path could eventually reach it
after one or more scatters.  The speedup provided by this solution is
substantial, but neglecting scattering can distort results, especially for
detector designs where the string distance is larger than the scattering
length.

p65

4.6.2 Total photon yield 

Figure 4.5 shows the total number of photons generated
by a cascade with respect to its primary particle’s energy. The dependence on
energy is linear over the considered range. A best fit over a range of energies
from 1 GeV to 100 TeV yields 

::

    Nphotons ≈ 1.74 * 10^7 * Eshower/100 GeV


4.6.3 Scattering table generation 

The most time-consuming process of the
simulation chain is the propagation of photons through the detector medium
including the simulation of scattering and ab-


p66,67

To exploit as many symmetries as possible, the shower is assumed to be
point-like, i.e. photons are only emitted from a single point in space which is
chosen to be at the origin during the simulation process. Additionally, showers
are assumed to have a rotational symmetry around their axis. In the following
description, this shower axis is arbitrarily defined to be the z-axis. During
the table generation process, it is sufficient to emit photons only with
directions in the x-z-plane, provided that an arbitrary rotation around the
z-axis is performed whenever such a photon is used later on.


p73

4.7 Adaptation of the cascade simulation approach to muons 

The method for shower photon simulation described in the previous section can only be used for
point-like light sources as it relies on the spherical symmetry of the source.
However, the basic idea of using a database of pre-propagated photons for fast
lookup can be extended to muons (or any other track-like light source), too.
This section describes a spatial segmentation that can be used for this
problem.


Common simulation tools for large volume neutrino detectors, A.Margiotta, ANTARES Collaboration
--------------------------------------------------------------------------------------------------

* https://www.sciencedirect.com/science/article/pii/S0168900212015197?via%3Dihub

* https://indico.cern.ch/event/143656/papers/1378074/files/1029-elsarticle-template-num_vlvnt11_rev.pdf

The goal of a neutrino telescope is the detection of astrophysical high-energy
neutrinos. Actually, most of the detectable Cherenkov light is due to the pas-
sage of high-energy atmospheric muons and of muons induced by atmospheric
neutrino interactions in the vicinity of the detector.

Cherenkov light emission and propagation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* "scattering tables" "photon field"
* individual photon propagation

MY THOUGHTS:

* these are not alternatives, the individual photon propgation is needed anyhow 
  to create the tables : and validate optimization "trickery"


CG Rendering Overview
-----------------------

* https://en.wikipedia.org/wiki/Ray_tracing_(graphics)

* (Appel 1968) ray casting 
* ray tracing 
* path tracing
* bi-directional path tracing 

**global illumination techniques**

* photon mapping 
* progressive photon mapping 

Cornell Box
-------------

* https://www.youtube.com/watch?v=Rk5nD8tt_W4
* Ray Tracing Essentials Part 5: Ray Tracing Effects

* 0:30 Which one is real ?

The Rendering Equation
-----------------------

* https://en.wikipedia.org/wiki/Rendering_equation
* ~/opticks_refs/Kajiya_1986_The_Rendering_Equation.pdf  


* Gareth Morgan on "The Rendering Equation"
* https://paperswelove.org/2015/video/gareth-morgan-the-rendering-equation/

  44:00 recursive integral eqn, relating    

* https://mathworld.wolfram.com/FredholmIntegralEquationoftheSecondKind.html

* https://www.fxguide.com/fxfeatured/the-art-of-rendering/

* https://blog.demofox.org/2016/09/21/path-tracing-getting-started-with-diffuse-and-emissive/


aka LTE : Light Transport Equation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://www.pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/The_Light_Transport_Equation.html


Good description of rendering equation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://cs.dartmouth.edu/~wjarosz/publications/dissertation/chapter2.pdf

* http://www.pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/The_Light_Transport_Equation.html





Computer graphics III – Rendering equation and its solution
---------------------------------------------------------------

* :google:`neumann solution of rendering equation`

* https://cgg.mff.cuni.cz/~jaroslav/teaching/2015-npgr010/slides/07%20-%20npgr010-2015%20-%20rendering%20equation.pdf
* https://cgg.mff.cuni.cz/~jaroslav/


* https://cgg.mff.cuni.cz/~jaroslav/papers/2018-mcvolrendering/index.htm


* https://graphics.stanford.edu/courses/cs348b-02/lectures/lecture14/renderingequation.pdf

* https://mycourses.aalto.fi/pluginfile.php/1204050/mod_resource/content/1/04_RenderingEquation.pdf

  * ~/opticks_refs/Jaakko_Lehtinen_04_RenderingEquation.pdf


* http://graphics.snu.ac.kr/class/graphics2011/materials/ch08_rendering02_renderingequation.pdf



Bi-directional path tracing
-----------------------------

* https://graphics.stanford.edu/courses/cs348b-03/papers/veach-chapter10.pdf


Monte Carlo Path Tracing : Ravi
----------------------------------

* https://www.youtube.com/watch?v=KCYroQVaARs

* sample all paths in the scene 


Adaptive Progressive Photon Mapping
-------------------------------------

* https://cg.ivd.kit.edu/publications/p2012/APPM_Kaplanyan_2012/APPM_Kaplanyan_2012.pdf

GPU Photon Mapping Using OptiX 3.5 + OpenGL compute shaders
-------------------------------------------------------------

* http://essay.utwente.nl/70708/1/Jimenez%20Kwast_MA_EEMCS.pdf

The rendering equation describes the distribution of radiance in a scene under
the assumption that the light has reached a state of equilibrium.

Path tracing – an algorithm first introduced in Kajiya’s rendering equation
paper [12] – is a variation of distribution ray tracing. The core concept is
that instead of tracing rays and generating multiple rays at each surface
intersection, a sample can be computed by evaluating the contribution of a
single path along which light may travel, starting from a pixel and ending at a
light source (with an arbitrary number of reflections in between). The result
is a flattened search space which turns the tree-like search space from
distribution ray tracing into a single path. This removes the explosiveness in
terms of the number of rays and reduces the computational costs of a single
sample. However, a much larger number of samples is needed per pixel and
ensuring a good distribution of reflection rays is considerably more difficult.



Tetrahedralization
--------------------

Light probe interpolation using tetrahedral tessellations
Robert Cupisz

* https://twvideo01.ubm-us.net/o1/vault/gdc2012/slides/Programming%20Track/Cupisz_Robert_Light_Probe_Interpolation.pdf

* 3D barycentric coordinates for tet 


Bounce Pics
-------------

* https://smerity.com/montelight-cpp/


Ray Tracing Roundup
---------------------

* http://www.realtimerendering.com/raytracing/roundup.html

* http://www.realtimerendering.com/raytracing.html


Overview
----------

* https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-overview/light-transport-ray-tracing-whitted

* https://cs.dartmouth.edu/~wjarosz/publications/dissertation/chapter2.pdf


SIGGRAPH 2019: GPU Ray Tracing for Film and Design : Introduction
-----------------------------------------------------------------

* https://developer.nvidia.com/siggraph/2019/video/sig910


RAY TRACING and other RENDERING METHODS, Andrey Lebrov
--------------------------------------------------------

* https://www.youtube.com/watch?v=LAsnQoBUG4Q

* rasterization : project 3d model polygons onto 2d image plane (entirely "cheating")
* ray casting : cast rays from camera, texture lookup (flat look, no shadows)
* ray tracing : again cast from camera, then cast more shadow rays to point lights (hard shadows)  
* path tracing : again cast thru pixels, but cast more rays (noise is a problem, eg Octane) 

Ray Tracing Essentials Part 6: The Rendering Equation, Eric Haines

* https://www.youtube.com/watch?v=AODo_RjJoUA


Path Tracing
--------------

* https://morgan3d.github.io/advanced-ray-tracing-course/path-tracing-review.pdf


“Pure” path tracing doesn’t work for point lights. There’s zero probability
that a ray hits a point light, and if it did, the radiance would be infinite.
So, we have to put direct illumination, shadow rays, and biradiance back in to
make point lights (vs. area lights) work. Sorry. We only use path tracing for
indirect light. BUT: The direct illumimation code is a lot faster than hoping a
random ray will hit the light source, and you already wrote and debugged it
anyway.


Volumetric Path Tracing
-------------------------

* https://en.wikipedia.org/wiki/Volumetric_path_tracing

* http://luthuli.cs.uiuc.edu/~daf/courses/Rendering/Papers/lafortune96rendering.pdf

Rendering Participating Media with Bidirectional Path Tracing
Eric P. Lafortune and Yves D. Willems
Paper presented at the 7th Eurographics Workshop on Rendering


Monte Carlo Ray Tracing from Scratch
-------------------------------------

* https://eriksvjansson.net/papers/mcrt.pdf
* ~/opticks_refs/Monte_Carlo_Raytracing_from_Scratch.pdf  


Path Tracing In Production
----------------------------

* https://jo.dreggn.org/path-tracing-in-production/2019/johannes_hanika.pdf
* https://jo.dreggn.org/path-tracing-in-production/2019/
* http://www.realtimerendering.com/raytracing/siggraph2019/Path_Tracing_in_Production_part_1.pdf

* ~/opticks_refs/Path_Tracing_in_Production_ptp-part1.pdf 


Monte Carlo methods for volumetric light transport simulation
In Computer Graphics Forum (Proceedings of Eurographics - State of the Art Reports), 2018

* https://cs.dartmouth.edu/~wjarosz/publications/novak18monte.html
* ~/opticks_refs/Monte_Carlo_Methods_for_Volumetric_Light_Transport_Simulation_novak18monte.pdf


References to teach CG / ray-tracing
----------------------------------------



* http://www.realtimerendering.com/raytracinggems/

collection of articles focused on ray tracing techniques for serious
practitioners. Like other "gems" books, it focuses on subjects commonly
considered too advanced for introductory texts, yet rarely addressed by
research papers.

* http://www.realtimerendering.com/raytracinggems/unofficial_RayTracingGems_v1.7.pdf

* ~/opticks_refs/unofficial_RayTracingGems_v1.7.pdf


Ray Tracing In One Weekend Book Series
----------------------------------------

* https://raytracing.github.io

* https://raytracing.github.io/books/RayTracingInOneWeekend.html#wherenext?

  Peter Shirley

* https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/

  Accelerated Ray Tracing in One Weekend in CUDA



A Framework for Transient Rendering
--------------------------------------

* http://giga.cps.unizar.es/~ajarabo/pubs/transientSIGA14/downloads/Jarabo_siga14.pdf

Time Sampling in Participating Media


Photon Simulations with Houdini. September 11, 2019
-------------------------------------------------------

* https://sergeneren.com/2019/09/11/photon-simulations/

Sources
---------


* http://graphics.stanford.edu/courses/Appel.pdf

* ~/opticks_refs/SIG19_OptiX_7_Main_Talk.pdf  

  NVIDIA Talk 112 pages 
  RTX ACCELERATED RAY TRACING WITH OPTIX

* p79 : samples per pixel comparison, 5/50/500/5000

* Disney's Practical Guide to Path Tracing
* https://www.youtube.com/watch?v=frLwRLS_ZR0



GEANT4 simulation of optical modules in neutrino telescopes
Christophe M.F. Hugon(INFN, Genoa)
Aug 16, 2015

* https://inspirehep.net/literature/1483400

* ~/opticks_refs/geant4_simulation_of_neutrino_telescopes_PoSICRC20151106.pdf



Invite
--------


::

    Dear Simon,

    This is Donglian Xu. I am an associate professor of physics at Shanghai Jiao
    Tong University, and a fellow at the Tsung-Dao Lee Institute. My primary
    research area is neutrino astronomy, and I am a member of the JUNO, IceCube and
    LHAASO collaborations. I learnt that you are also a collaborator of JUNO,
    really hope to meet you in person in our next collaboration meeting :)

    As Tao (in cc) probably mentioned to you, we are organizing a simulation
    workshop aiming to optimize the design of the next-generation neutrino
    telescopes. Those are giant detectors encompassing O(10) km^3 transparent
    interaction medium such as ice or water. We need to simulate the passage of
    energetic (TeV-PeV) charged particles (induced by high-energy neutrinos)
    through ice/water and collect the Cherenkov photons emitted. The fundamental
    photon sensors are digital optical modules (DOMs) with PMTs housed in
    transparent and pressure-resistant glass vessels. They form a very sparse
    sensor array with typical horizontal spacing ~ 100m, and vertical spacing ~
    20m. One of the main challenges is to trace every Cherenkov photon emitted
    through the medium till they are absorbed or hit a DOM. As one can imagine, it
    is extremely time consuming to trace every single photon in such a huge
    detector volume!

    Our PhD student Fan Hu (in cc) is driving this simulation effort currently, and
    he found out about your great work on ray-tracing with Opticks for JUNO. We
    were wondering if we can make use of your nice techniques of ray-tracing in
    neutrino telescopes. Therefore, we are cordially inviting you to give a 1-hour
    lecture on Opticks in our upcoming simulation workshop. 

    More info about the scope of the workshop can be found on the indico page (access code: tdli2020): 

    https://indico-tdli.sjtu.edu.cn/event/238/overview

    Tao told us you are in UK now, so we've tentatively scheduled your talk to be
    on ~16:00 of 8.13 Beijing time (9:00am London time). Please let us know if you
    can accept our invitation to speak via ZOOM. If the answer is positive, we will
    be more than happy to reallocate any time slot that works best for you.

    Looking forward to hearing back from you soon, and do take good care! 

    With best regards,

    Donglian 


Dear Donglian, 

I accept your invitation. 

My provisional title:

   Opticks: GPU photon simulation via NVIDIA OptiX ;  
   Applied to neutrino telescope simulations ?  

Your tentative schedule for next Thursday 13th is OK for me.
My network connection is often poor, so it would be better 
for my slides to be shared from your end.  

Simon



::

    Hi Tao, 

    > Fan Hu, a PhD student from PKU, is working with Donglian Xu from SJTU on the
    > simulation of neutrino telescope in the deep sea. He is interested in your work
    > and hope to invite you to give an online seminar.  Do you have time? If you
    > have time, I will tell them.  BTW: the LHAASO experiment is also interested in
    > your Opticks. Maybe we could arrange another seminar at IHEP.  Tao

    I think a small “workshop” type meeting bringing together the people who are actually 
    working on simulation from KM3Net, LHAASO etc.. would be more productive than just 
    giving my seminar again to a general audience.
    For better communication everyone attending should give a short presentation 
    on how they currently simulate and how they would like to use Opticks 
    or similar to accelerate it.

    Optical simulation for deep underwater neutrino telescopes 
    like KM3Net or Baikal GVD inevitably needs to take an indirect approach
    due to the extreme numbers of photons being impossible to store.
    Instead of storing photons I guess it will be necessary 
    to develop a way to progressively accumulate into data structures 
    such as progressive photon maps (eg kd-tree based)
    or light fields at chosen positions relative to the cosmics 
    and the strings of PMTs.

    The Opticks approach could be adapted to accumulating into such 
    data structures. Basically instead of collecting 
    photon parameter "samples" binned probability distributions 
    are collected.

    Designing the light field/photon map data structure and 
    a way to accumulate into it and demonstrating that it can answer the 
    physics questions that need to be answered are the major requirements.

    I would start by searching for recent developments from
    the graphics community in such data structures. 

    For an introduction to global illumination and photon mapping 
    For background I recommend a classic book :

       "Realistic Image Synthesis Using Photon Mapping"
       Henrik Wann Jensen
       http://graphics.stanford.edu/~henrik/papers/book/

    However the static photon map (using a kd-tree) described 
    is probably not the thing to do. 
    Instead investigate "progressive photon mapping"
    techniques from the graphics community.  
    Computer vision research has developed light field structures 
    that might also be worth investigating.

    One paper that describes progressive photon mapping:
       https://www.sciencedirect.com/science/article/pii/S0038092X15000559

    The thesis of Eric Veach 
       Robust Monte Carlo Methods for Light Transport Simulation 
       http://graphics.stanford.edu/papers/veach_thesis/

    is a good starting point for getting familiar with 
    graphics community developments in light transport and getting
    used to their terminology, eg "bi-directional path tracing"
    and "global illumination".

    Simon




:google:`GPU photon mapping with OptiX`
------------------------------------------

Progressive Photon Mapping on GPUs
Stian Aaraas Pedersen

* ~/opticks_refs/Progressive_Photon_Mapping_on_GPUs_Stian_Pedersen_52105235.pdf
 


Progressive Photon Mapping: A Probabilistic Approach
Claude Knaus and Matthias Zwicker University of Bern, Switzerland

* https://www.cs.umd.edu/~zwicker/publications/PPMProbabilistic-TOG11.pdf
* ~/opticks_refs/Progressive_Photon_Mapping_a_Probabalistic_Approach_PPMProbabilistic-TOG11.pdf



* https://github.com/immiao/PhotonMapping
* https://web.cs.wpi.edu/~emmanuel/courses/cs563/write_ups/zackw/photon_mapping/PhotonMapping.html



Principles of Light Field Imaging: Briefly revisiting 25 years of research
----------------------------------------------------------------------------

* https://en.wikipedia.org/wiki/Light_field
* https://hal.inria.fr/hal-01377379/file/main.pdf

* Plenoptic Function



Volumetric Photon Mapping
---------------------------

* https://github.com/jacklv123/Volumetric-photon-mapping/blob/master/main.cpp


The Beam Radiance Estimate for Volumetric Photon Mapping
----------------------------------------------------------

* https://cs.dartmouth.edu/~wjarosz/publications/jarosz08beam-tech.pdf


Rendering Course
------------------------------------------------------------------------------

* https://www.cg.tuwien.ac.at/courses/Rendering/VU.SS2020.html

This course will teach you how to write a physically correct and unbiased
renderer. You will learn how to accelerate ray-triangle intersection using
acceleration structures, the math and physics behind rendering, how to compute
high-dimensional integrals using Monte Carlo methods, and how to apply all that
to implement the recursive path-tracing algorithm. 

We will also introduce other important rendering algorithms like bidirectional
path tracing, Metropolis light transport, photon mapping and others.
Furthermore we will talk about material models, participating media,
HDR/tonemapping and some state-of-the-art papers in the rendering domain. 

At the end of the course students should be familiar with common techniques in
rendering and find their way around the current state-of-the-art of the field.
Furthermore the exercises should deepen the attendees' understanding of the
basic principles of light transport and enable them to write a rendering
program themselves.


Path Tracing : with nice focus on code "Implementing the rendering equation"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://www.cg.tuwien.ac.at/courses/Rendering/2020/slides/07_path_tracing.pdf



* https://www.scratchapixel.com/lessons/3d-basic-rendering/global-illumination-path-tracing/global-illumination-path-tracing-practical-implementation


Production Volume Rendering
-----------------------------


* https://graphics.pixar.com/library/ProductionVolumeRendering/paper.pdf
* ~/opticks_refs/Pixar_sigraph2017_ProductionVolumeRendering_paper.pdf



* http://www.pbr-book.org/3ed-2018/contents.html

* http://www.pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/The_Light_Transport_Equation.html



Familiar Tracking Approaches
------------------------------

Tracking approaches employ Russian roulette and rejection sampling strategies
to decide on a single type of collision being sampled instead of trying to
estimate all of them at the same time.



Efficient Monte Carlo Methods for Light Transport in Scattering Media, Wojciech Jarosz
------------------------------------------------------------------------------------------

* Ph.D. dissertation, UC San Diego, September 2008.


Found this thesis via 
* https://en.wikipedia.org/wiki/Photon_mapping

* https://cs.dartmouth.edu/~wjarosz/publications/dissertation/
* https://cs.dartmouth.edu/~wjarosz/publications/dissertation/dissertation-web.pdf
* ~/opticks_refs/wjarosz-dissertation-web.pdf


* p63 The Henyey-Greenstein Phase Function
* p70 Stochastic Methods 

For example, homogeneous media with a high scattering albedo can be modeled
accurately using a *diffusion approximation* [Stam, 1995; Jensen et al., 2001b],
which leads to very efficient rendering algorithms. Premoze et al. [2004],
under the assumption that the medium is tenuous and strongly forward
scattering, use a path integral formulation to derive efficient rendering
algorithms. Sun et al. [2005] render single scattering in real time, but
without shadowing effects.


Jos Stam. Multiple scattering as a diffusion process. In Patrick M. Hanrahan
and Werner Purgath- ofer, editors, Rendering Techniques ’95, Eurographics,
pages 41–50. Springer-Verlag Wien New York, 1995. 3, 70

* http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.20.9501&rep=rep1&type=pdf
* ~/opticks_refs/multiple_scattering_as_a_diffusion_process_jos_stam_10.1.1.20.9501.pdf

* p72 Chapter 5 : Radiance Caching in Participating Media



Wojciech Jarosz
----------------

* https://cs.dartmouth.edu/~wjarosz/index.html


Photon Mapping Course
----------------------

* https://web.archive.org/web/20110607074737/http://www.cs.princeton.edu/courses/archive/fall02/cs526/papers/course43sig02.pdf
* ~/opticks_refs/henrik_wann_jensen_photon_mapping_course43sig02.pdf


p26 : Volume Photon Map 

p27 

Another argument that is perhaps even more important is the fact that a
balanced kd-tree can be represented using a heap-like data-structure
[Sedgewick92] which means that explicitly storing the pointers to the sub-trees
at each node is no longer necessary. 
(Array element 1 is the tree root, and element i has element 2i as
left child and element 2i + 1 as right child.) This can lead to considerable
savings in memory when a large number of photons is used.


p28 : balanced kd-tree  (binary space partioning to speed up spatial search) 



p34 : The radiance estimate in a participating medium


Henrik Wann Jensen
Realistic Image Synthesis using Photon Mapping
AK Peters, 2001







A Framework for Transient Rendering
-------------------------------------


* http://giga.cps.unizar.es/~ajarabo/pubs/transientSIGA14/


Femto-Photography: Capturing and Visualizing the Propagation of Light
-----------------------------------------------------------------------

* http://giga.cps.unizar.es/~ajarabo/pubs/femtoSIG2013/


Key points to convey
---------------------

* computer graphics : try to re-purpose techniques  
* too many resources online ! 

Need to explain the field of CG !

* global illumination 
* path tracing 
* ray tracing 

Pointers for what to follow 


* http://www.realtimerendering.com/raytracing/Ray%20Tracing%20in%20a%20Weekend.pdf
* https://github.com/petershirley/raytracinginoneweekend
* https://raytracing.github.io

* https://raytracing.github.io/books/RayTracingInOneWeekend.html

* https://github.crookster.org/raytracing-iow-in-cpp-cuda-and-optix/


* https://github.com/trevordblack/OptixInOneWeekend
* https://github.com/joaovbs96/OptiX-Path-Tracer



