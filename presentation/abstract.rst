Abstract
=========


GTC
----

::

    Opticks : Optical Photon Simulation for High Energy Physics with OptiX 

    Opticks is an open source project that integrates high performance GPU 
    ray tracing from NVIDIA OptiX with existing Geant4 toolkit based simulations.
    Advantages of separate optical photon simulation and    
    the approaches developed to integrate it with the general Geant4
    particle simulation are presented. Approaches to minimize overheads
    arising from split are shown.
    Challenges included bringing complex geometries with wavelength
    dependent material and surface properties to the GPU.
    Techniques for visualisation of photon propagations with
    interactive time scrubbing and history selection using OpenGL/OptiX/Thrust
    interoperation and geometry shaders are described.
    Results and demonstrations are shown for the photomultiplier based 
    Daya Bay and JUNO Neutrino detectors. 
    Extrapolation of observed timings with test geometries to multi-GPU workstation
    core counts suggests performance sufficient for optical photon processing time to 
    become effectively zero compared to total simulation time is within reach.



LeCosPA
---------

::

    Opticks : Optical Photon Simulation for Particle Physics using GPU Accelerated Ray Tracing

    A detailed understanding of the generation and propagation of optical photons is vital 
    to the design, operation and analysis of photomultiplier based neutrino detectors such as
    the Daya Bay and JUNO experiments. Detailed detector simulations handling many millions 
    of optical photons per interaction are used to develop the understanding required 
    to make precision measurements. Traditional serial approaches using the Geant4 simulation 
    toolkit suffer from an optical photon simulation problem where photon handling provides 
    a processing bottleneck and constitutes a CPU memory resource limitation.

    I will present Opticks, https://bitbucket.org/simoncblyth/opticks, an open source project 
    that is my solution to the optical photon simulation problem. 
    Opticks works together with Geant4, replacing the Geant4 optical photon simulation 
    with an equivalent massively parallel optical simulation implemented on the GPU 
    and accelerated with the NVIDIA OptiX ray tracing engine.
    Optical photon simulation performance with Opticks is extrapolated to exceed 1000x Geant4 
    with workstation GPU machines. Optical photon processing time becomes effectively zero
    and CPU memory constraints from the optical photons are eliminated. Also moving the simulation
    to the GPU has enabled unprecedented in the field visualizations to be developed. 

    Although developed in the context of Geant4 based neutrino detector simulations the 
    techniques used by Opticks are potentially applicable to any simulation that is
    limited by optical photon processing.  


Qingdao
----------

::

    Opticks : Optical Photon Simulation for Particle Physics using GPU Accelerated Ray Tracing,
    Focussing on Tools and Techniques

    A detailed understanding of the generation and propagation of optical photons is vital 
    to the design, operation and analysis of photomultiplier based neutrino detectors such as JUNO. 
    Detailed detector simulations handling many millions of optical photons per interaction are 
    needed to develop the understanding required to make precision measurements. 
    Traditional serial approaches using the Geant4 simulation toolkit suffer from an optical 
    photon simulation problem where photon handling provides a processing bottleneck 
    and constitutes a CPU memory resource limitation.

    I will present Opticks an open source project 
    that solves the optical photon simulation problem by replacing the Geant4 optical photon simulation 
    with an equivalent massively parallel GPU simulation implemented using the NVIDIA OptiX ray tracing engine.
    Optical photon simulation performance with Opticks is extrapolated to exceed 1000x Geant4 
    with workstation GPU machines. Optical photon processing time becomes effectively zero
    and CPU memory constraints from the optical photons are eliminated. 

    In order to provide directly actionable information my presentation covers techniques 
    such as Monte Carlo sampling and gives examples of the use of data science tools 
    such as NumPy, CUDA and Thrust.

    









