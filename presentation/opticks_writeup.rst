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


Introduction
-------------

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


Ray Tracing Refs
-------------------

* https://mediatech.aalto.fi/~timo/



Ray Tracing and NVIDIA OptiX
-----------------------------------

Ray tracing is an active area of computer science research 


NVIDIA OptiX
--------------






