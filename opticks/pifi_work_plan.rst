PIFI Work Plan in Five Stages 
================================================

The primary goal of my stay at IHEP is to bring Opticks into active production usage 
within the JUNO Collaboration, providing accelerated simulation to all JUNO physicists.
Opticks directly ports existing optical physics implementations and detector geometry 
to the GPU, thus excellent agreement between CPU and GPU simulations can be achieved.
Attaining this level of agreement however requires extensive validation work 
that I plan to complete working together with JUNO detector experts 
during my stay. 

The work required to realise the potential of Opticks within JUNO 
can be divided into five stages:

1. Opticks integration with the JUNO framework 
2. validation using aligned random sequence simulations 
3. JUNO geometry debugging 
4. small scale deployment and production testing 
5. wider deployment 


Opticks Integration with JUNO 
---------------------------------

Opticks is integrated with Geant4 by modifications to the scintillation and
Cerenkov processes, instead of generating photons in a loop the “genstep”
parameters are collected, including the number of photons to generate and a
line segment along which to generate them and any other parameters needed for
the photon generation. These gensteps are copied to the GPU 
photon generation and propagation with photons that hit photomultipliers 
being returned into the standard Geant4 hit collections.

Some JUNO specific glue code is required to configure an embedded Opticks
instance which receives gensteps and returns parameters of photons that 
hit photomultiplier tubes.  

Validation using aligned random sequence simulations 
----------------------------------------------------

Opticks includes CUDA programs for the optical physics processes of scattering, 
absorption, scintillator reemission and boundary processes.  
These reimplementations are all based upon Geant4, and require careful validation 
by comparison against Geant4.

In order to rapidly discover deviations between the Opticks and
Geant4 simulations I have developed an aligned mode of operation 
where the same pre-generated photons and random number sequences 
are used as inputs to both simulations. 
By careful matching of the random number consumption 
by the two simulations it has been possible to arrange a near perfect match 
with every scatter, absorption, reflection or refraction occuring 
with matched positions, times and polarizations when simulating simple
geometries. Some small float precision level differences of up to 10^-5 ns or mm 
between photon parameters of the simulations are observed.

Step-by-step alignment with the expectation of perfect matching to within 
the precision allows deviations to be discovered as soon as they occur, 
and once a match has been achieved provides the best possible demonstration 
of validity.

I plan to attempt to expand code alignment to photon generation by scintillation 
and Cerenkov processes and scintillator reemission. This would allow aligned 
operation from input gensteps in addition to input photons.
The self contained nature of the photon generation code 
suggests that it will be straightforward to align. Aligning reemission 
is more technically challenging due to the different arrangements 
of the Geant4 and Opticks reemission code.

JUNO Geometry Debugging
--------------------------

Most of the development effort of Opticks has concerned
the translation of detector geometry into GPU appropriate forms
enabling application of accelerated ray-geometry intersection.  

Geometry issues such as overlapping or touching volumes create
ambiguities which Opticks ray tracing arbitrarily chooses between.  
As increasingly complex geometries are tested building up to the 
full JUNO geometry deviations from geometry ambiguities are expected.
It will be necessary to work with geometry authors to fix such issues, 
or workarounds such as very small volume shifts or scaling will need to be
developed to remove the ambiguities.

Currently the neck of the JUNO 20 inch PMTs is modelled by subtraction of a torus primitive.
Also the JUNO calibration guide tube is modelled by a torus with very large major radius and
small minor radius. Finding ray-torus intersection entails the solution of quartic equations 
which are prone to numerical stability problems. Efforts to mitigate or avoid these
issues via alternative modelling will be required.     

Small Scale Deployment and Production Testing 
----------------------------------------------

Performance testing of JUNO/Opticks simulation installations 
with a variety of photon loads and hardware configurations spanning 
from multi-GPU servers down to single GPU desktops and laptops 
are needed to establish limitations and hardware guidelines. 
This testing experience is also required in order to develop 
automated splitting of GPU launches on systems that require this.

Guidelines for minimum and optimum machine configurations 
for various use cases including production running and development 
need to be established to assist collaborators making hardware purchases.

Wider deployment 
----------------

Successful installation of large software packages with many dependencies 
requires expertise to fix installation issues.  Efforts to simplify 
installation through installer scripts and documentation are necessary 
but typically not sufficient to achieve a wide deployment.

Investigations of containerized distribution techniques may 
circumvent many installation issues and enable deployments
onto locally managed GPU machines across the world as well as 
GPU instances provided by cloud services.  

High performance geometry and event visualization is a natural bonus 
resulting from moving optical processing to the GPU. 
Opticks uses this performance in an interactive simulated event and geometry 
display application developed for simulation debugging. 
Additional features could be developed to widen the usefulness of this
tool, for example to simplify the creation of event display animations.
These animations can be used to convey the principals of JUNO detector operation 
in an engaging manner that is accessible to general audiences. 

Developing a Geant4 advanced example demonstrating Opticks with JUNO geometry
for potential inclusion with the Geant4 distribution would provide an excellent
way to introduce the benefits of Opticks to a wide audience.

